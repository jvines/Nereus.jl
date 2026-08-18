# Tests for the O'Neil 2019 observation-based prior.

@testset "O'Neil obs_prior" begin

    function _make_setup_with_obs_prior(obs_prior_flag::Bool)
        instruments = InstrumentConfig(rv=["X"])
        relast = RelAstromData(
            t = [55000.0, 56000.0, 57000.0],
            ra_off = [10.0, 12.0, 8.0],
            dec_off = [-5.0, -3.0, -7.0],
            ra_err = [1.0, 1.0, 1.0],
            dec_err = [1.0, 1.0, 1.0],
        )
        hgca = HGCAData(epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.,0.,0.), pmdec=(0.,0.,0.),
            sigma_pmra=(0.5,0.02,0.03), sigma_pmdec=(0.4,0.02,0.03),
            plx=10.0, plx_err=0.1, hip_id=1)
        data = Data(t_rv=[55000.0, 55100.0], rv=[0.0,0.0], rv_err=[1.0,1.0],
                    relastrom=relast, hgca=hgca)
        params = Params(max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=1.0,
            parametrization=ParametrizationConfig(mass=:a_driven, obs_prior=obs_prior_flag),
            priors=Dict("a_k1" => LogUniformPrior(0.5, 1000.0)),
        )
        return data, params
    end

    @testset "obs_prior=false → astrom_log_likelihood matches sum of relAST + HGCA" begin
        data, params = _make_setup_with_obs_prior(false)
        theta = Theta(params)
        set_param!(theta, "n_p", 1.0)
        set_param!(theta, "a_k1", 5.0)
        set_param!(theta, "M_sec_k1", 0.001)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", π/2)
        set_param!(theta, "Omega_k1", 0.0)
        set_param!(theta, "plx", 10.0)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)
        ll_total = astrom_log_likelihood(theta, data)
        ll_rel = relastrom_log_likelihood(theta, data)
        ll_hg  = hgca_log_likelihood(theta, data)
        @test ll_total ≈ ll_rel + ll_hg
    end

    @testset "obs_prior=true adds non-zero correction" begin
        _, params_off = _make_setup_with_obs_prior(false)
        _, params_on  = _make_setup_with_obs_prior(true)
        @test params_off.config.parametrization.obs_prior == false
        @test params_on.config.parametrization.obs_prior  == true

        # Build identical theta in both
        function _set(theta)
            set_param!(theta, "n_p", 1.0)
            set_param!(theta, "a_k1", 5.0)
            set_param!(theta, "M_sec_k1", 0.001)
            set_param!(theta, "sesinw_k1", 0.0)
            set_param!(theta, "secosw_k1", 0.0)
            set_param!(theta, "Mo_k1", 0.0)
            set_param!(theta, "inc_k1", π/2)
            set_param!(theta, "Omega_k1", 0.0)
            set_param!(theta, "plx", 10.0)
            set_param!(theta, "gamma_X", 0.0)
            set_param!(theta, "sigma_X", 1.0)
        end
        # Need fresh data for each; reuse the relast/hgca
        data, _ = _make_setup_with_obs_prior(false)
        theta_off = Theta(params_off); _set(theta_off)
        theta_on  = Theta(params_on);  _set(theta_on)
        ll_off = astrom_log_likelihood(theta_off, data)
        ll_on  = astrom_log_likelihood(theta_on,  data)
        # The obs_prior contributes a non-zero finite term
        @test isfinite(ll_off)
        @test isfinite(ll_on)
        @test ll_on != ll_off
    end

    @testset "obs_based_log_prior is finite at sensible params" begin
        data, params = _make_setup_with_obs_prior(true)
        theta = Theta(params)
        set_param!(theta, "n_p", 1.0)
        set_param!(theta, "a_k1", 5.0)
        set_param!(theta, "M_sec_k1", 0.001)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", π/2)
        set_param!(theta, "Omega_k1", 0.0)
        set_param!(theta, "plx", 10.0)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)
        @test isfinite(obs_based_log_prior(theta, data))
    end

    @testset "obs_based_log_prior returns 0 with no relAST" begin
        instruments = InstrumentConfig(rv=["X"])
        hgca = HGCAData(epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.,0.,0.), pmdec=(0.,0.,0.),
            sigma_pmra=(0.5,0.02,0.03), sigma_pmdec=(0.4,0.02,0.03),
            plx=10.0, plx_err=0.1, hip_id=1)
        # No relastrom
        data = Data(t_rv=[55000.0, 55100.0], rv=[0.0,0.0], rv_err=[1.0,1.0], hgca=hgca)
        params = Params(max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=1.0,
            parametrization=ParametrizationConfig(mass=:a_driven, obs_prior=true))
        theta = Theta(params)
        set_param!(theta, "n_p", 1.0)
        set_param!(theta, "a_k1", 5.0)
        set_param!(theta, "M_sec_k1", 0.001)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", π/2)
        set_param!(theta, "Omega_k1", 0.0)
        set_param!(theta, "plx", 10.0)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)
        @test obs_based_log_prior(theta, data) == 0.0
    end
end
