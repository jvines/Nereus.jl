# Resumable downloads and in-process zip extraction (no network: a loopback
# HTTP server stands in for the archives' hosts).

using Sockets
using p7zip_jll

# Minimal HTTP/1.1 server. `range` toggles Range support; `cut` is the number of
# body bytes after which the next GET's connection is dropped (once).
function _serve(payload::Vector{UInt8}; range::Bool = true, cut::Ref{Int} = Ref(-1),
                stall::Ref{Int} = Ref(-1), chunk::Int = 0, pause::Float64 = 0.0)
    server = listen(IPv4(127, 0, 0, 1), 0)
    port = Int(getsockname(server)[2])
    log = Tuple{String,String}[]
    task = @async while isopen(server)
        sock = try accept(server) catch; break end
        @async try
            head = String[]
            while true
                l = readline(sock)
                isempty(strip(l)) && break
                push!(head, l)
            end
            method = split(head[1])[1]
            rng = nothing
            for l in head[2:end]
                startswith(l, "Range:") && (rng = strip(l[7:end]))
            end
            push!(log, (method, rng === nothing ? "" : rng))
            n = length(payload)
            first_byte = 0
            status = "200 OK"
            extra = ""
            if range && rng !== nothing
                first_byte = parse(Int, match(r"bytes=(\d+)-", rng).captures[1])
                status = "206 Partial Content"
                extra = "Content-Range: bytes $first_byte-$(n - 1)/$n\r\n"
            end
            body = payload[first_byte+1:end]
            write(sock, "HTTP/1.1 $status\r\nContent-Length: $(length(body))\r\n" *
                        "Accept-Ranges: bytes\r\n$extra" *
                        "Connection: close\r\n\r\n")
            if method == "GET"
                if stall[] >= 0           # send a few bytes, then hang (once)
                    k = stall[]; stall[] = -1
                    write(sock, body[1:min(k, length(body))]); flush(sock)
                    sleep(30)
                elseif chunk > 0          # slow but steady
                    for i in 1:chunk:length(body)
                        write(sock, body[i:min(i + chunk - 1, end)]); flush(sock)
                        sleep(pause)
                    end
                elseif cut[] >= 0
                    k = cut[]; cut[] = -1
                    write(sock, body[1:min(k, length(body))])
                else
                    write(sock, body)
                end
            end
            flush(sock)
        catch
        finally
            close(sock)
        end
    end
    return (url = "http://127.0.0.1:$port/f.bin", server = server, log = log)
end

@testset "resumable download + in-process unzip" begin
    payload = rand(UInt8, 300_000)
    mktempdir() do dir
        # --- plain download, nothing left behind
        s = _serve(payload)
        dest = joinpath(dir, "a.bin")
        Nereus._download_resumable(s.url, dest; backoff = 0.01)
        @test read(dest) == payload
        @test !isfile(dest * ".part")
        close(s.server)

        # --- interrupted mid-body, then resumed with a Range request
        cut = Ref(100_000)
        s = _serve(payload; cut = cut)
        dest = joinpath(dir, "b.bin")
        Nereus._download_resumable(s.url, dest; backoff = 0.01)
        @test read(dest) == payload
        @test any(l -> l[1] == "GET" && startswith(l[2], "bytes=100000-"), s.log)
        close(s.server)

        # --- a .part left by an earlier run is continued, not restarted
        s = _serve(payload)
        dest = joinpath(dir, "c.bin")
        write(dest * ".part", payload[1:12_345])
        Nereus._download_resumable(s.url, dest; backoff = 0.01)
        @test read(dest) == payload
        @test any(l -> l[1] == "GET" && l[2] == "bytes=12345-", s.log)
        close(s.server)

        # --- server without Range support: restart from zero, still correct
        s = _serve(payload; range = false)
        dest = joinpath(dir, "d.bin")
        write(dest * ".part", payload[1:12_345])
        Nereus._download_resumable(s.url, dest; backoff = 0.01)
        @test read(dest) == payload
        @test !any(l -> l[1] == "GET" && l[2] != "", s.log)
        close(s.server)

        # --- an oversized stale .part is discarded
        s = _serve(payload)
        dest = joinpath(dir, "e.bin")
        write(dest * ".part", rand(UInt8, length(payload) + 10))
        Nereus._download_resumable(s.url, dest; backoff = 0.01)
        @test read(dest) == payload
        close(s.server)

        # --- stalled transfer: aborted after the stall window, then resumed
        stall = Ref(50_000)
        s = _serve(payload; stall = stall)
        dest = joinpath(dir, "g.bin")
        t = @elapsed Nereus._download_resumable(s.url, dest; stall_timeout = 2,
                                                backoff = 0.01)
        @test read(dest) == payload
        @test 1.5 < t < 20
        @test any(l -> l[1] == "GET" && startswith(l[2], "bytes=50000-"), s.log)
        close(s.server)

        # --- slow but steady: longer than the stall window, still completes
        slow = rand(UInt8, 60_000)
        s = _serve(slow; chunk = 10_000, pause = 0.7)
        dest = joinpath(dir, "h.bin")
        t = @elapsed Nereus._download_resumable(s.url, dest; stall_timeout = 2,
                                                attempts = 1)
        @test read(dest) == slow
        @test t > 3                 # exceeded the 2 s window in total
        close(s.server)

        # --- unreachable host: fails after the attempts, no final file
        dest = joinpath(dir, "f.bin")
        @test_throws ErrorException Nereus._download_resumable(
            "http://127.0.0.1:1/x", dest; attempts = 2, backoff = 0.01, probe_timeout = 5.0)
        @test !isfile(dest)

        # --- zip member extraction keeps the internal path (cache layout)
        src = joinpath(dir, "src"); mkpath(joinpath(src, "ResRec", "H001"))
        write(joinpath(src, "ResRec", "H001", "H001234.d"), "iad rows\n")
        write(joinpath(src, "ResRec", "H001", "other.d"), "not wanted\n")
        zip = joinpath(dir, "bundle.zip")
        cd(src) do
            run(pipeline(`$(p7zip_jll.p7zip()) a -tzip $zip ResRec`, stdout = devnull))
        end
        out = joinpath(dir, "out")
        @test Nereus._unzip_member(zip, "ResRec/H001/H001234.d", out)
        @test read(joinpath(out, "ResRec", "H001", "H001234.d"), String) == "iad rows\n"
        @test !isfile(joinpath(out, "ResRec", "H001", "other.d"))
        @test !Nereus._unzip_member(zip, "ResRec/H001/missing.d", joinpath(dir, "out2")) ||
              !isfile(joinpath(dir, "out2", "ResRec", "H001", "missing.d"))
    end
end
