using DataFrames
using CSV

include("LDA.jl")

cd("/Users/markkeller/Documents/UMD/Summer2019/julia_inference_exercises/lda")
docs = CSV.read("data/sim-movie-review-data-docs.tsv", delim='\t')
topics = CSV.read("data/sim-movie-review-data-topics.tsv", delim='\t')

n_topics = 3

# Store each document as an array of words
texts = Array{Array{String,1},1}()
# Construct a set to store the unique words
vocab = Set(String[])
# Iterate over each value of the Words column,
# where each value is a string `text` containing the words of the document
for text in docs.Words
    words = split(text)
    push!(texts, words)
    union!(vocab, words)
end


# Construct document-term matrix
n_vocab_words = length(vocab)
n_docs = length(texts)
word_to_token = Dict{String,Int32}(zip(vocab, 1:n_vocab_words))
token_to_word = Dict{Int32,String}(zip(1:n_vocab_words, vocab))

# Construct the token-based corpus
corpus = Array{Array{Int32,1},1}(undef, n_docs)
for d in 1:n_docs
    words = texts[d]
    n_words = length(words)
    corpus[d] = Array{Int32,1}(undef, n_words)
    for n in 1:n_words
        word = words[n]
        token = word_to_token[word]
        corpus[d][n] = token
    end
end

LDA(corpus, token_to_word, random_state=10, K=n_topics, max_iter=100)
