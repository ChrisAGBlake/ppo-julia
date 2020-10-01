module Utils
export connect_to_s3, s3_bucket, sync_with_s3

using PyCall
boto3 = pyimport("boto3")

const s3_bucket = "gombot-prestart-057919834666-us-east-2"

function connect_to_s3()
    s3 = boto3.client("s3")
    return s3
end

function sync_with_s3(s3, dir, exp_name)

    # get the keys of the checkpoints in s3 
    keys = Array{String}(undef, 10000)
    n = 0
    response = s3.list_objects(
        Bucket=s3_bucket, Prefix=string(exp_name, "/models/")
    )
    if get(response, "Contents", nothing) !== nothing
        for obj in response["Contents"]
            n += 1
            keys[n] = obj["Key"]
        end
        while response["IsTruncated"]
            response = s3.list_objects(
                Bucket=s3_bucket, 
                Prefix=string(exp_name, "/models/"),
                Marker=keys[n]
            )
            for obj in response["Contents"]
                n += 1
                keys[n] = obj["Key"]
            end
        end
    end
    if n > 0
        keys = view(keys, 1:n)

        # get the checkpoints already downloaded
        present = readdir(dir)

        # step through the keys and download models that aren't present
        for i = eachindex(keys)
            # check if this checkpoint is already present
            name = match(r"model\d_.*", keys[i])
            if name === nothing
                name = match(r"boat\d_.*", keys[i])
            end
            if name !== nothing
                download = true
                name = name.match
                r = Regex("$name")
                for j = eachindex(present)
                    m = match(r, present[j])
                    if m !== nothing
                        download = false
                        break
                    end
                end
                if download
                    s3.download_file(s3_bucket, keys[i], string(dir, name))
                end
            end
        end
    end

    return n
end

end
