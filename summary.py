import pandas as pd
import numpy as np
import sys


def get_summary(filename):
    with open(filename, "r", newline='') as ifile:
        compile_df = pd.read_csv(ifile)
    # Use RangeId as the DataFrame index
    compile_df = compile_df.set_index(compile_df.pop("RangeId").astype(np.int32))
    compile_df = compile_df.rename(
        columns={
            "Start (ns)": "StartNs",
            "Duration (ns)": "DurNs",
            "DurChild (ns)": "DurChildNs",
            "DurNonChild (ns)": "DurNonChildNs",
        }
    ).drop(columns=["End (ns)", "PID", "TID", "Lvl", "NameTree"])
    # "TSL:XlaCompile:#module=" is the top range
    for compile_id in compile_df.index[compile_df["Name"].str.startswith("TSL:XlaCompile:#module=")]:
        # Trailing : excludes the top-level XlaCompile range
        compile_name = compile_df.loc[compile_id, "Name"]
        compile_time = compile_df.loc[compile_id, "DurNs"]
        # every range is assigned 1 id, and stack is shown as id1:id2:id3
        mask = compile_df["RangeStack"].str.startswith(f":{compile_id}:")
        attributed_time = compile_df.loc[mask, ("Name", "DurNonChildNs")]
        # Aggregate over some details
        def shorten_name(long_name):
            # XlaPass, XlaPassPipeline, etc.
            # name = long_name.split(":")[1]
            names = long_name.split(":")
            if len(names) >= 2:
                name = names[1]
            else:
                name = long_name
            return {"XlaPassPipeline": "XlaPass"}.get(name, name)
            # return long_name
        attributed_time.loc[:, "ShortName"] = attributed_time["Name"].apply(shorten_name)
        aggregated = attributed_time.groupby("ShortName").agg({"DurNonChildNs": "sum"})
        aggregated.sort_values(by=["DurNonChildNs"], ascending=False, inplace=True)
        aggregated_total = aggregated["DurNonChildNs"].sum()
        aggregated.loc[:, "DurNonChildFrac"] = aggregated["DurNonChildNs"] / aggregated_total
        aggregated.loc[:, "DurNonChildCumFrac"] = aggregated["DurNonChildFrac"].cumsum()
        # Show the largest contributors that add up to >X of the total
        fraction = 0.99
        top_culprits = aggregated[aggregated["DurNonChildCumFrac"] < fraction]
        print(f"Compilation: {compile_name}: {aggregated_total / 1000000000}")
        for row in top_culprits.itertuples():
            print(f" {row.Index} {row.DurNonChildFrac:.2%} {row.DurNonChildNs / 1000000000:.4}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 summary.py [filename]")
    args = sys.argv[1]
    get_summary(args)