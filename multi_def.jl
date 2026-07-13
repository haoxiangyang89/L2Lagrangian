# definition of data structure

struct nodeType
    time :: Int64
    index :: Int64
    predecessor :: Int64
    d :: Dict{Int64,Any}
    successor :: Array{Int64,1}
    probability :: Array{Float64,1}
    sign :: Dict{Int64,Any}
end

mutable struct problemData
    I :: Array{Int64,1}
    K :: Array{Any,1}

    p_up :: Dict{Int64,Any}
    p_norm :: Dict{Int64,Any}
    p_down :: Dict{Int64,Any}

    s :: Dict{Int64,Any}
    b :: Dict{Int64,Any}
    h :: Dict{Int64,Any}
    Capacity :: Float64

    xlb :: Dict{Int64,Any}
    xub :: Dict{Int64,Any}

    f :: Array{Any,1}
end