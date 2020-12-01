## Here we collect the results gained from the computations made in the scripts
# stored in the `cluster scripts`-folder:
# For a range of `F`-parameter values (F=2.5:0.1:6) we analyzed embeddings
# obtained from traditional time delay embedding, from PECUZAL and from MCDTS, by
# computing RQA-quantifiers.
# Run the script `comm_lorenz96_cluster.jl` before running this script

using DelimitedFiles

writedlm("./application/artificial data/Lorenz96/Results/chosen_time_series.csv", t_idx)


varnames = ["tau_tde", "optimal_d_tde", "RQA_tde", "L_tde",
    "tau_pec", "ts_pec", "optimal_d_pec", "RQA_pec", "L_pec",
    "tau_MCDTS", "ts_MCDTS", "optimal_d_mcdts", "RQA_mcdts", "L_mcdts"]

for i = 1:length(varnames)
    writestr = "./application/artificial data/Lorenz96/Results/"*varnames[i]*".csv"
    data = []
    for j = 1:length(results)
        push!(data,results[j][i])
        writedlm(writestr, data)
    end
end
