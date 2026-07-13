
# Create box plot for number of products = 3
using Plots, StatsPlots, DataFrames

# Calculate statistics
mean_UB = mean(VbarList)
median_UB = median(VbarList)
LB = Vstar_UB

# Create the box plot using StatsPlots
# Create a DataFrame for the box plot
df = DataFrame(ObjectiveValue=VbarList, Group=fill(3, length(VbarList)))

# Create the box plot
p = @df df boxplot(:Group, :ObjectiveValue,
    xlabel="Number of products", 
    ylabel="Objective value",
    title="Box Plot for Number of Products = 3",
    legend=:topright,
    xticks=([3], ["3"]),
    ylims=(minimum([VbarList; LB]) * 1.1, maximum([VbarList; LB]) * 1.1),
    size=(600, 400)
)

# Add mean UB (green triangle)
scatter!([3], [mean_UB], 
        marker=:utriangle, 
        color=:green, 
        markersize=8, 
        label="Mean UB")

# Add median UB (orange horizontal line)
hline!([median_UB], 
       color=:orange, 
       linestyle=:solid, 
       linewidth=2, 
       label="Median UB")

# Add LB (blue circle)
scatter!([3], [LB], 
        marker=:circle, 
        color=:blue, 
        markersize=8, 
        label="LB")

# Display the plot
display(p)

# Save the plot
savefig(p, "boxplot_products_3.png")
println("Plot saved as boxplot_products_3.png")
println("Mean UB: ", mean_UB)
println("Median UB: ", median_UB)
println("LB: ", LB)

# Store results in orig_results dictionary
orig_results = Dict("LB_list" => LB_list, "time_list" => time_list)

# Plot LB_list as a line plot
p1 = plot(orig_results["LB_list"][2:length(orig_results["LB_list"])],
    xlabel="Iteration",
    ylabel="Lower Bound (LB)",
    title="Lower Bound Progress",
    linewidth=2,
    color=:blue,
    legend=false,
    size=(600, 400),
    grid=true
)
display(p1)
savefig(p1, "lb_progress.png")
println("LB plot saved as lb_progress.png")

# Plot cumulative time
cumulative_time = cumsum(orig_results["time_list"])
p2 = plot(cumulative_time,
    xlabel="Iteration",
    ylabel="Cumulative Time (seconds)",
    title="Cumulative Time Progress",
    linewidth=2,
    color=:red,
    legend=false,
    size=(600, 400),
    grid=true
)
display(p2)
savefig(p2, "cumulative_time.png")
println("Cumulative time plot saved as cumulative_time.png")