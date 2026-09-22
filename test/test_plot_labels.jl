# Axis-label LaTeX for the science posterior plots.
#
# The bug this guards against: `_sci_sym` used to append `_{k}` to a base that
# already carried its own underscore, so a chain column named `M_sec_k1` became
# `M_sec_{1}` — a DOUBLE SUBSCRIPT. MathTeXEngine renders that as three loose
# atoms (M_s·e·c_1) for `M_sec`, and refuses to parse it outright for `K_A_k1`
# / `K_B_k1`, which took the whole SB2 figure down with it. A second defect in
# the same function did greek substitution by SUBSTRING, which turned `rho_s`
# into `\mathrm{\rho\,s}` (command stranded inside \mathrm, subscript gone) and
# silently missed capital `Omega`.
#
# The assertions below are deliberately of two kinds:
#   * PROPERTY — every parameter name the layout can produce must yield LaTeX
#     that MathTeXEngine actually parses, and a unicode label with no raw `_`.
#     This is what catches the NEXT parameter someone adds with an underscore.
#   * ANCHOR — exact expected strings for the names that were broken, so a
#     regression names itself instead of showing up as a vibe.

using Nereus, Test
using CairoMakie: Makie

const _MTE = Makie.MathTeXEngine

# MathTeXEngine's own `showerror` is broken (it throws UndefVarError on :tex),
# so never let the exception escape to be printed — only its existence matters.
_parses(tex::AbstractString) = try
    _MTE.texparse(String(tex))
    true
catch
    false
end

# Instrument labels really do carry underscores (HARPS_DRS) and `+` (the
# group-sharing label "HARPS_DRS+SERVAL", src/model.jl:1422).
const _INSTR = ["HARPS_DRS", "HARPS_SERVAL", "HIRES", "CORALIE98", "ESPRESSO",
                "FEROS", "UVES", "PFS", "TESS", "NEID", "CARMENES_VIS",
                "CARMENES_NIR", "HARPS_DRS+SERVAL"]

# Every per-planet base the unit registry knows about, plus the ones model.jl
# pushes that have no unit entry. Driving this off `_SCI_UNITS` means a new
# parameter added to the registry is covered here for free.
function _planet_names()
    bases = collect(keys(Nereus._SCI_UNITS))
    append!(bases, ["M_sec", "K_A", "K_B", "sesinw", "secosw", "Omega", "lambda",
                    "inc", "r1", "r2", "rr", "b", "Tp", "Tc", "Mo", "a"])
    return ["$(b)_k$k" for b in unique(bases) for k in 1:3]
end

# Shared / stellar / systemic columns (src/model.jl:1476-1703).
const _SHARED = ["n_p", "rho_s", "dvdt", "d2vdt2", "plx", "M_pri", "i_star",
                 "v_sin_i_star", "sigma_ccf", "beta_p", "f_light",
                 "ttv_k1_t3", "ttv_k2_t12"]

# Per-instrument / per-noise columns (src/model.jl:1499-1605, noise models).
_instrument_names() = [p * "_" * i for i in _INSTR for p in
                       ("gamma", "sigma", "jitter", "offset", "dilution", "q1",
                        "q2", "u1", "u2", "C", "sigma_as", "gd_beta", "phot_c0",
                        "phot_c1", "tomo_alpha", "tomo_sigma_line", "tomo_ell_v",
                        "gp_log_S0", "gp_sigma", "Cdot_bis")]

const _IND_FLOOR = ["ind_floor_period", "ind_floor_lambda_e", "ind_floor_lambda_p",
                    "ind_floor_halpha_amp", "ind_floor_bis_jit",
                    "ind_floor_log_rhk_amp", "ind_floor_fwhm_jit",
                    "ind_floor_bisector_span_amp"]

const _ALL_NAMES = vcat(_planet_names(), _SHARED, _instrument_names(), _IND_FLOOR)

@testset "science plot axis labels" begin

    # -----------------------------------------------------------------
    # PROPERTY: the LaTeX x-axis symbol always parses.
    # Failures are collected rather than asserted one by one so a regression
    # prints the offending NAMES, not 500 anonymous passes and one red dot.
    @testset "every symbol is parseable LaTeX" begin
        @test !isempty(_ALL_NAMES)
        unparseable = [nm => Nereus._sci_sym(nm) for nm in _ALL_NAMES
                       if !_parses(Nereus._sci_sym(nm))]
        @test isempty(unparseable)
    end

    # PROPERTY: symbol + unit, i.e. the full `_sci_label` payload, also parses.
    # Units come from the registry itself, so a new unit is covered for free.
    @testset "every symbol × unit combination parses" begin
        units = unique(vcat("", "mas", [u for (u, _) in values(Nereus._SCI_UNITS)]))
        @test "M_sun" in units          # the unit that used to render as "M sun"
        full(nm, u) = (ul = Nereus._sci_unit_latex(u);
                       ul == "" ? Nereus._sci_sym(nm) :
                                  Nereus._sci_sym(nm) * "\\;(" * ul * ")")
        unparseable = [(nm, u) for nm in _ALL_NAMES for u in units
                       if !_parses(full(nm, u))]
        @test isempty(unparseable)
    end

    # PROPERTY: the unicode y-axis label never shows a raw underscore. It is a
    # plain String (MathTeXEngine cannot line-break a LaTeXString), so nothing
    # would error — an underscore would just be drawn literally on the figure.
    @testset "no raw underscore in unicode labels" begin
        underscored = [nm => Nereus._sci_sym_u(nm) for nm in _ALL_NAMES
                       if occursin('_', Nereus._sci_sym_u(nm))]
        @test isempty(underscored)
    end

    # -----------------------------------------------------------------
    # ANCHOR: the names that were actually broken.
    @testset "double-subscript regressions" begin
        # M_sec parsed but rendered garbled; K_A / K_B did not parse at all.
        @test Nereus._sci_sym("M_sec_k1") == "M_{\\mathrm{sec},1}"
        @test Nereus._sci_sym("K_A_k1")   == "K_{A,1}"
        @test Nereus._sci_sym("K_B_k2")   == "K_{B,2}"
        for nm in ("M_sec_k1", "K_A_k1", "K_B_k2")
            @test _parses(Nereus._sci_sym(nm))
        end
        # The shape that caused it: never a `_` outside a brace group after the
        # first subscript opens.
        @test !occursin("_{1}_", Nereus._sci_sym("M_sec_k1"))
    end

    @testset "greek is matched per token, not per substring" begin
        @test Nereus._sci_sym("rho_s")      == "\\rho_\\star"
        @test Nereus._sci_sym("beta_p")     == "\\beta_{p}"
        @test Nereus._sci_sym("Omega_k1")   == "\\Omega_{1}"   # capital, was "Omega"
        @test Nereus._sci_sym_u("Omega_k1") == "Ω₁"
        # A greek command must never end up inside \mathrm{}, where it loses the
        # subscript and renders detached.
        for nm in ("rho_s", "beta_p", "gd_beta_TESS", "tomo_alpha_HARPS")
            @test !occursin(r"\\mathrm\{[^}]*\\(alpha|beta|rho|sigma|lambda)", Nereus._sci_sym(nm))
        end
    end

    @testset "stellar and sibling symbols" begin
        @test Nereus._sci_sym("i_star")       == "i_{\\star}"
        @test Nereus._sci_sym("v_sin_i_star") == "v\\sin i_\\star"
        @test Nereus._sci_sym_u("i_star")       == "i⋆"
        @test Nereus._sci_sym_u("v_sin_i_star") == "v sin i⋆"
        @test Nereus._sci_sym_u("rho_s")        == "ρ⋆"
        @test Nereus._sci_sym("M_pri")        == "M_{\\mathrm{pri}}"
        # esinw/ecosw and inc were missed while their siblings sesinw/secosw
        # already had proper symbols.
        @test Nereus._sci_sym("esinw_k1") == "e\\,\\sin\\omega_{1}"
        @test Nereus._sci_sym("ecosw_k1") == "e\\,\\cos\\omega_{1}"
        @test Nereus._sci_sym("inc_k1")   == "i_{1}"
        @test Nereus._sci_sym_u("esinw_k1") == "e·sinω₁"
        @test Nereus._sci_sym_u("inc_k1")   == "i₁"
    end

    @testset "units" begin
        @test Nereus._sci_unit_latex("M_sun") == "M_\\odot"   # was \mathrm{M\,sun}
        @test Nereus._sci_unit_latex("M_earth") == "M_\\oplus"
        @test Nereus._sci_unit_latex("") == ""
    end

    # The unicode path spells instrument names the same way the LaTeX path does
    # (`\mathrm{HARPS\,DRS}` draws as "HARPS DRS"), so the two axes agree.
    @testset "instrument names render consistently" begin
        @test Nereus._sci_sym_u("sigma_HARPS_DRS")    == "σ HARPS DRS"
        @test Nereus._sci_sym_u("gamma_HARPS_DRS")    == "γ HARPS DRS"
        @test Nereus._sci_sym_u("sigma_as_HARPS_DRS") == "σ as HARPS DRS"
        @test Nereus._sci_sym("gamma_HARPS_DRS") == "\\gamma_{\\mathrm{HARPS\\,DRS}}"
    end

    # Previously-correct labels that must not drift while the fallbacks change.
    @testset "unchanged symbols" begin
        @test Nereus._sci_sym("P_k1")  == "P_{1}"
        @test Nereus._sci_sym("K_k1")  == "K_{1}"
        @test Nereus._sci_sym("w_k2")  == "\\omega_{2}"
        @test Nereus._sci_sym("Tc_k1") == "T_{\\mathrm{c},1}"
        @test Nereus._sci_sym("rr_k1") == "(R_p/R_\\star)_{1}"
        @test Nereus._sci_sym("ind_floor_period") == "P_{\\mathrm{floor}}"
        @test Nereus._sci_sym_u("P_k1") == "P₁"
    end
end
