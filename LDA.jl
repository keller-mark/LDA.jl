using Random
using Distributions

include("LDAUtils.jl")
using .LDAUtils

#=
LDA generative model:
- For each topic k ∈ {1, ..., K} draw a multinomial distribution β[k]
    from a Dirichlet distribution with parameter λ.
    (Draw a distribution over words for each topic.)
- For each document d ∈ {1, ..., M} draw a multinomial distribution θ[d]
    from a Dirichlet distribution with parameter α
    (Draw a distribution over topics for each document.)
- For each word position n ∈ {1, ..., N}, select a hidden topic z[n]
    from the multinomial distribution parameterized by θ
    (Draw a topic for each word position.)
- Choose the observed word w[n] from the distribution β[z[n]]
    (Draw a word from the topic selected for the word position.)


Deriving variational inference for LDA
    references:
        - https://www.cs.colorado.edu/~jbg/teaching/CSCI_5622/19a.pdf
        - https://www.youtube.com/watch?v=Dv86zdWjJKQ
--------------------------------------
Joint distribution:
    p(θ, z, w | α, β) = \prod_{d}{ p(θ[d] | α) } \prod_{n}{ p(z[d,n] | θ[d])*p(w[d,n] | β, z[d,n]) }
                      = probability of choosing the topic mixings for the document given prior α
                        * probability of choosing the topic assignment for the word position given topic mixings
                        * probability of choosing the word given the topics and the topic assignment for the word

    notice:
        p(θ[d] | α) is a draw from dirichlet (per document topic mixings)
        p(z[d,n] | θ[d]) is a draw from multinomial (word-topic assignment given topic mixings)
        p(w[d,n] | β, z[d,n]) is a draw from multinomial (word probability given topic)

Variational distribution:
    motivation: posterior is intractable for large n

    create a variational distribution over the latent variables q(z | ν)
    goal: find the settings of ν so that q is close to the posterior
        measture the closeness of distributions using KL divergence
        if KL == 0 then two distributions are equal

    for a particular topic:
        q(θ,z) = q(θ | γ)*q(z | ϕ)
               = \prod_{d}{ q(θ[d] | γ[d]) } \prod_{n}{ q(z[d,n] | ϕ[d,n]) }

               where γ[d] is the variational document distribution over topics
                     ϕ[d,n] is the variational token distribution over topic assignments

    for all topics:
        q(β,θ,z) = q(β | λ)*q(θ | γ)*q(z | ϕ)
                 = \prod_{k}{ q(β[k] | λ[k]) } \prod_{d}{ q(θ[d] | γ[d]) } \prod_{n}{ q(z[d,n] | ϕ[d,n]) }


    ELBO (evidence lower bound)
    L(γ,ϕ; α,β) = E_q( \log{p(θ | α)} )
                  + E_q( \log{p(z | θ)} )
                  + E_q( \log{p(w | z,β)} )
                  - E_q( \log{q(θ)} )
                  - E_q( \log{q(z)} )

    omitted: derivation of the complete objective function L

Update for ϕ
    ϕ[n,i] ∝ β[i,v]*exp( Ψ(γ[i]) - Ψ(\sum_{j}{ γ[j] } )

Update for γ
    γ[i] = α[i] + \sum_{n}{ ϕ[n,i] }

Update for β
    β[i,j] ∝ \sum_{d} \sum_{n}{ ϕ[d,n,i]*w^j[d,n] }

Classical algorithm (mean-field variational inference)
    1. Randomly initialize variational parameters (can't be uniform)
    2. For each iteration
        2.1. For each document, update γ and ϕ
        2.2. For corpus, update β
        2.3. Compute L for diagnostics
    3. Return expectation of variational parameters for solution to latent variables

Stochastic algorithm (stochastic variational inference)
    1. Subsample data from larger dataset
    2. Infer local structure of subsampled data
    3. Update idea of the global structure from the local structure
    4. Repeat

    more specifically:
        - initialize λ randomly
        - set ρ_t appropriately
        - repeat the following:
            sample j ~ Uniform(1,...,n)
            set local parameter ϕ
            set intermediate global parameter \hat{λ}
            set global parameter λ = (1 - ρ_t)*λ + ρ_t*\hat{λ}

    for LDA:
        - sample a document
        - estimate the local variational parameters using the current topics
        - form intermediate topics from those local parameters
        - update topics as a weighted average of intermediate and current topics



Stochastic optimization
    - Replace the gradient with cheaper noisy estimates of the gradient
    - Guaranteed to converge to a local optimum
    - Requires unbiased gradient
    - Requires that step size constant ρ_t follows Robbins-Monroe conditions

Logistics:
    α is vector of length K
        (prior for drawing topic mixings per-document)
    λ is vector of length V
        (prior for drawing word probabilities per-topic)
    β is matrix of size K x V
        (topics)
    θ is matrix of size D x K
        (topic mixings)
    z is vector of size N where N is number of words in the document
        (per-document word topic assignments)
    w is vector of size N where N is number of words in the document
        (per-document words)

    γ is matrix of size D x K
    ϕ is array of size D x N x K where N is number of words in each document
=#

function LDA(corpus :: Array{Array{Int32,1},1}, token_to_word :: Dict{Int32,String}; random_state :: Int64, K :: Int64, λ :: Union{Array{Float32,1}, Nothing}=nothing, α :: Union{Array{Float32,1}, Nothing}=nothing, max_iter=1e3)
    Random.seed!(random_state)

    D = size(corpus, 1) # number of documents in corpus
    V = length(token_to_word) # number of words in vocabulary

    # Initialize α if not provided as a parameter
    if α != nothing
        assert(length(α) == K)
    else
        α = fill(5/K, K)
    end

    # Initialize λ if not provided as a parameter
    if λ != nothing
        assert(length(λ) == V)
    else
        λ = fill(V/100, V)
    end

    # 1.1 Randomly initialize topics
    β = Array{Float64,2}(undef, K, V)
    for k in 1:K
        β[k] = Dirichlet(λ)
    end

    # 1.2 Randomly initialize variational parameters
    γ = Array{Float64,2}(undef, D, K) # array of size D x K
    ϕ = Array{Array{Float64,2},1}(undef, D) # array of size D x N x K
    for d in 1:D
        n_words = length(corpus[d])
        ϕ[d] = Array{Float64,2}(undef, n_words, K)
    end

    # 2.0 Iterate
    for i in 1:max_iter
        # 2.1 For each document
        for d in 1:D
            # 2.1.1 Update ϕ and γ
            N = length(corpus[d]) # number of words
            for n in 1:N
                v = token_to_word[corpus[d][n]]
                for k in 1:K
                    ϕ[d][n][k] = β[k][v]*exp( Ψ(γ[d][k]) - Ψ(sum(γ[d])) )
                end
            end
            γ[d] = α + sum(ϕ[d], dims=1)
        end

        # 2.2 For corpus, update β
        for k in 1:K
            for v in 1:V
                # TODO: fix the line below
                β[k][v] = sum( sum( ϕ[d][n][k]*corpus[d][v] ) )
            end
        end

        # 2.3 Compute L for diagnostics

    end

    # 3.0 Return expectation of variational parameters


end
