#=
Approximations of the gamma and digamma functions.

References:
- https://en.wikipedia.org/wiki/Gamma_function
- https://en.wikipedia.org/wiki/Digamma_function
- https://commons.apache.org/proper/commons-math/javadocs/api-3.6/src-html/org/apache/commons/math3/special/Gamma.html
=#

module LDAUtils
        # Euler-Mascheroni constant.
        const GAMMA = MathConstants.γ
        # Lanczos approximation constant.
        const LANCZOS_G = (607.0 / 128.0)
        # Maximum allowed numerical error.
        const DEFAULT_EPSILON = 10e-15
        # Lanczos coefficients.
        const LANCZOS = Float64[
                0.99999999999999709182,
                57.156235665862923517,
                -59.597960355475491248,
                14.136097974741747174,
                -0.49191381609762019978,
                .33994649984811888699e-4,
                .46523628927048575665e-4,
                -.98374475304879564677e-4,
                .15808870322491248884e-3,
                -.21026444172410488319e-3,
                .21743961811521264320e-3,
                -.16431810653676389022e-3,
                .84418223983852743293e-4,
                -.26190838401581408670e-4,
                .36899182659531622704e-5,
        ]

        # Cache √2π
        const SQRT_TWO_PI = 2.506628274631000502

        # Limits for switching algorithm in digamma
        const C_LIMIT = 49.0
        const S_LIMIT = 1e-5

        # Constants for the computation of double invGamma1pm1(double)
        const INV_GAMMA1P_M1_A0 = .611609510448141581788E-08
        const INV_GAMMA1P_M1_A1 = .624730830116465516210E-08
        const INV_GAMMA1P_M1_B1 = .203610414066806987300E+00
        const INV_GAMMA1P_M1_B2 = .266205348428949217746E-01
        const INV_GAMMA1P_M1_B3 = .493944979382446875238E-03
        const INV_GAMMA1P_M1_B4 = -.851419432440314906588E-05
        const INV_GAMMA1P_M1_B5 = -.643045481779353022248E-05
        const INV_GAMMA1P_M1_B6 = .992641840672773722196E-06
        const INV_GAMMA1P_M1_B7 = -.607761895722825260739E-07
        const INV_GAMMA1P_M1_B8 = .195755836614639731882E-09
        const INV_GAMMA1P_M1_P0 = .6116095104481415817861E-08
        const INV_GAMMA1P_M1_P1 = .6871674113067198736152E-08
        const INV_GAMMA1P_M1_P2 = .6820161668496170657918E-09
        const INV_GAMMA1P_M1_P3 = .4686843322948848031080E-10
        const INV_GAMMA1P_M1_P4 = .1572833027710446286995E-11
        const INV_GAMMA1P_M1_P5 = -.1249441572276366213222E-12
        const INV_GAMMA1P_M1_P6 = .4343529937408594255178E-14
        const INV_GAMMA1P_M1_Q1 = .3056961078365221025009E+00
        const INV_GAMMA1P_M1_Q2 = .5464213086042296536016E-01
        const INV_GAMMA1P_M1_Q3 = .4956830093825887312020E-02
        const INV_GAMMA1P_M1_Q4 = .2692369466186361192876E-03
        const INV_GAMMA1P_M1_C = -.422784335098467139393487909917598E+00
        const INV_GAMMA1P_M1_C0 = .577215664901532860606512090082402E+00
        const INV_GAMMA1P_M1_C1 = -.655878071520253881077019515145390E+00
        const INV_GAMMA1P_M1_C2 = -.420026350340952355290039348754298E-01
        const INV_GAMMA1P_M1_C3 = .166538611382291489501700795102105E+00
        const INV_GAMMA1P_M1_C4 = -.421977345555443367482083012891874E-01
        const INV_GAMMA1P_M1_C5 = -.962197152787697356211492167234820E-02
        const INV_GAMMA1P_M1_C6 = .721894324666309954239501034044657E-02
        const INV_GAMMA1P_M1_C7 = -.116516759185906511211397108401839E-02
        const INV_GAMMA1P_M1_C8 = -.215241674114950972815729963053648E-03
        const INV_GAMMA1P_M1_C9 = .128050282388116186153198626328164E-03
        const INV_GAMMA1P_M1_C10 = -.201348547807882386556893914210218E-04
        const INV_GAMMA1P_M1_C11 = -.125049348214267065734535947383309E-05
        const INV_GAMMA1P_M1_C12 = .113302723198169588237412962033074E-05
        const INV_GAMMA1P_M1_C13 = -.205633841697760710345015413002057E-06

        struct NumberIsTooSmallException <: Exception
                wrong_val::Float64
                min_val::Float64
                bound_is_allowed::Bool
        end
        struct NumberIsTooLargeException <: Exception
                wrong_val::Float64
                max_val::Float64
                bound_is_allowed::Bool
        end

        # Returns the Lanczos approximation used to compute the gamma function.
        function lanczos(x :: Float64)
                sum = 0.0
                i = length(LANCZOS)
                while i > 1
                        # note the 1-indexed julia arrays
                        sum += LANCZOS[i] / (x + (i-1))
                        i -= 1
                end

                return sum + LANCZOS[1]
        end

        # Returns the value of (1 / Γ(1 + x) - 1) for -0.5 <= x <= 1.5
        function inv_gamma_1pm1(x :: Float64)
                if x < -0.5
                        throw(NumberIsTooSmallException(x, -0.5, true))
                end
                if x > 1.5
                        throw(NumberIsTooLargeException(x, 1.5, true))
                end

                ret = NaN64
                t = x
                if x > 0.5
                        t = x - 0.5
                end
                t -= 0.5
                if t < 0.0
                        a = INV_GAMMA1P_M1_A0 + t * INV_GAMMA1P_M1_A1

                        b = INV_GAMMA1P_M1_B8
                        b = INV_GAMMA1P_M1_B7 + t * b
                        b = INV_GAMMA1P_M1_B6 + t * b
                        b = INV_GAMMA1P_M1_B5 + t * b
                        b = INV_GAMMA1P_M1_B4 + t * b
                        b = INV_GAMMA1P_M1_B3 + t * b
                        b = INV_GAMMA1P_M1_B2 + t * b
                        b = INV_GAMMA1P_M1_B1 + t * b
                        b = 1.0 + t * b

                        c = INV_GAMMA1P_M1_C13 + t * (a / b)
                        c = INV_GAMMA1P_M1_C12 + t * c
                        c = INV_GAMMA1P_M1_C11 + t * c
                        c = INV_GAMMA1P_M1_C10 + t * c
                        c = INV_GAMMA1P_M1_C9 + t * c
                        c = INV_GAMMA1P_M1_C8 + t * c
                        c = INV_GAMMA1P_M1_C7 + t * c
                        c = INV_GAMMA1P_M1_C6 + t * c
                        c = INV_GAMMA1P_M1_C5 + t * c
                        c = INV_GAMMA1P_M1_C4 + t * c
                        c = INV_GAMMA1P_M1_C3 + t * c
                        c = INV_GAMMA1P_M1_C2 + t * c
                        c = INV_GAMMA1P_M1_C1 + t * c
                        c = INV_GAMMA1P_M1_C + t * c

                        if x > 0.5
                                ret = t * c / x
                        else
                                ret = x * ((c + 0.5) + 0.5)
                        end
                else
                        p = INV_GAMMA1P_M1_P6
                        p = INV_GAMMA1P_M1_P5 + t * p
                        p = INV_GAMMA1P_M1_P4 + t * p
                        p = INV_GAMMA1P_M1_P3 + t * p
                        p = INV_GAMMA1P_M1_P2 + t * p
                        p = INV_GAMMA1P_M1_P1 + t * p
                        p = INV_GAMMA1P_M1_P0 + t * p

                        q = INV_GAMMA1P_M1_Q4
                        q = INV_GAMMA1P_M1_Q3 + t * q
                        q = INV_GAMMA1P_M1_Q2 + t * q
                        q = INV_GAMMA1P_M1_Q1 + t * q
                        q = 1.0 + t * q

                        c = INV_GAMMA1P_M1_C13 + (p / q) * t
                        c = INV_GAMMA1P_M1_C12 + t * c
                        c = INV_GAMMA1P_M1_C11 + t * c
                        c = INV_GAMMA1P_M1_C10 + t * c
                        c = INV_GAMMA1P_M1_C9 + t * c
                        c = INV_GAMMA1P_M1_C8 + t * c
                        c = INV_GAMMA1P_M1_C7 + t * c
                        c = INV_GAMMA1P_M1_C6 + t * c
                        c = INV_GAMMA1P_M1_C5 + t * c
                        c = INV_GAMMA1P_M1_C4 + t * c
                        c = INV_GAMMA1P_M1_C3 + t * c
                        c = INV_GAMMA1P_M1_C2 + t * c
                        c = INV_GAMMA1P_M1_C1 + t * c
                        c = INV_GAMMA1P_M1_C0 + t * c

                        if x > 0.5
                                ret = (t / x) * ((c - 0.5) - 0.5)
                        else
                                ret = x * c
                        end
                end

                return ret
        end

        #=
        The gamma function.
        =#
        function Γ(x :: Float64)
                ret = NaN64

                if x == round(x) && x <= 0.0
                        return ret
                end

                abs_x = abs(x)
                if abs_x <= 20.0
                        if x >= 1.0
                                prod = 1.0
                                t = x
                                while t > 2.5
                                        t -= 1.0
                                        prod *= t
                                end
                                ret = prod / (1.0 + inv_gamma_1pm1(t - 1.0))
                        else
                                prod = x
                                t = x
                                while t < -0.5
                                        t += 1.0
                                        prod *= t
                                end
                                ret = 1.0 / (prod * (1.0 + inv_gamma_1pm1(t)))
                        end
                else
                        y = abs_x + LANCZOS_G + 0.5
                        gamma_abs = SQRT_TWO_PI / abs_x * y^(abs_x + 0.5) * exp(-y) * lanczos(abs_x)

                        if x > 0.0
                                ret = gamma_abs
                        else
                                ret = -π / (x * sin(π * x) * gamma_abs)
                        end
                end
                return ret
        end

        #=
        The digamma function.
        =#
        function Ψ(x :: Float64)
                if isnan(x) || isinf(x)
                        return x
                end

                if x > 0 && x <= S_LIMIT
                        return (-GAMMA - 1 / x)
                end

                if x >= C_LIMIT
                        inv = 1.0 / (x * x)
                        return (log(x) - 0.5 / x - inv * ((1.0 / 12) + inv * (1.0 / 120 - inv / 252)))
                end

                return (Ψ(x+1) - 1 / x)
        end

        export Γ, Ψ
end
