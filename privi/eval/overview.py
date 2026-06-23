from collections import defaultdict
import json
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import logging
from multiprocessing.pool import ThreadPool
import yaml
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# See README.md for how to use the functions in this script


##### METRIC AGGREGATORS ####
# Used for combining the metrics of several head runs into one aggregate score

def mean():
    def fn(x):
        return {"mean": x.mean()}
    return fn

def noagg():
    def fn(x):
        return {"vals": x.tolist()} 
    return fn

###### RUN AGGREGATOR #######
# Aggregate information from several head runs into one dataframe row, using one of the metric aggregators above

def last(metrics_to_report, metric_agg_fn, prefix=""):
    """
    Use the last logged checkpoint of the runs as final checkpoint, used if the dataset only has a val set and no test set.
    """
    def fn(df):
        def per_run(sdf):
            epoch = sdf.groupby("split")["epoch"].max().min()
            out = {"epoch": epoch}
            for m in metrics_to_report:
                split, metric = m.split("_", 1)
                results_df = sdf.query("epoch == @epoch and split == @split and metric == @metric")
                if results_df.empty:
                    out[m] = float("nan")
                else:
                    if len(results_df) > 1:
                        logger.warning(f"Too many hits for {epoch=}, {split=}, {metric=}")
                    out[m] = results_df.sort_values("step").iloc[-1]["val"]
            return pd.Series(out)
        
        rdf = df.groupby(["run", "head"], sort=False).apply(per_run)

        out = {f"{prefix}epoch": list(rdf["epoch"]), f"{prefix}n": len(rdf["epoch"])}
        for m in metrics_to_report:
            out.update({f"{prefix}{m}_{k}": v for k, v in metric_agg_fn(rdf[m]).items()})
        
        return pd.Series(out)
    return fn

def best(decision_metric, metrics_to_report, metric_agg_fn, prefix="", max_epoch=None):
    """"
    Choose the checkpoints with best val scores as final checkpoints, used if the dataset has separate val and test sets.
    """
    def fn(df):
        def per_run(sdf):
            split, metric = decision_metric.split("_", 1)
            sub = sdf.query("split == @split and metric == @metric")
            if max_epoch is not None:
                sub = sub[sub.epoch <= max_epoch]
            if sub.empty:
                return pd.Series({"epoch": float("nan"), decision_metric: float("nan"), **{m: float('nan') for m in metrics_to_report}})
            
            if sub["val"].isna().all(): # all NaNs in sub
                # Fallback to last epoch
                epoch_at_max = sdf.groupby("split")["epoch"].max().min()
            else:
                idx = sub["val"].idxmax()
                epoch_at_max = sdf.loc[idx, "epoch"]
            out = {"epoch": epoch_at_max}
            for m in metrics_to_report + [decision_metric]:
                split, metric = m.split("_", 1)
                results_df = sdf.query("epoch == @epoch_at_max and split == @split and metric == @metric")
                if results_df.empty:
                    out[m] = float("nan")
                else:
                    if len(results_df) > 1:
                        logger.warning(f"Too many hits for {epoch_at_max=}, {split=}, {metric=}")
                    out[m] = results_df.sort_values("step").iloc[-1]["val"]
            return pd.Series(out)
        
        rdf = df.groupby(["run", "head"], sort=False).apply(per_run)

        out = {f"{prefix}epoch": list(rdf["epoch"]), f"{prefix}n": len(rdf["epoch"]), f"{prefix}max_epoch": df.epoch.max()}
        for m in metrics_to_report + [decision_metric]:
            out.update({f"{prefix}{m}_{k}": v for k, v in metric_agg_fn(rdf[m]).items()})
        
        return pd.Series(out)
    return fn


##### MAIN ENTRYPOINTS #####

def agg(df, cfgs, display_cfgs: Dict[str, str], agg_fns):

    if not isinstance(agg_fns, (list, tuple)):
        agg_fns = [agg_fns]

    grouped = df.groupby(["proj", "group"], sort=False)

    cfg_df = {}

    for (proj, group), _ in grouped:
        cfg = cfgs[(proj, group)]
        row = {}

        for short, key in display_cfgs.items():
            parts = key.split(".")
            sub = cfg
            # traverse to the parent of the leaf
            for p in parts:
                if p not in sub:
                    logger.warning(f"No such nested key: {'.'.join(parts[:parts.index(p)+1])}")
                    sub = ""
                    break
                sub = sub[p]
            row[short] = sub

        cfg_df[(proj, group)] = row

    cfg_df = pd.DataFrame.from_dict(cfg_df, orient="index")
    cfg_df.index = pd.MultiIndex.from_tuples(cfg_df.index, names=["proj", "group"])

    dfs = [cfg_df]

    for agg_fn in agg_fns:
        dfs.append(grouped.apply(agg_fn))


    df = pd.concat(dfs, axis=1, verify_integrity=True)

    return df


def extract(projects_dirs : List[str], runs_to_merge : Optional[List[List[str]]]=None, n_threads=20) -> pd.DataFrame:
    """
    Long Form Extractor
    - give list of projects as parameter, optionally a dict which run groups should be merged
    - extract a long-form df with the following columns:
        - project, group_base, group_date, run_name, run_no, head_no, split, epoch, step, metric, value
    - get a dict of dicts with key: (group_base, group_date) -> cfg dict of this group
    - issue a warning when several runs in the same group have different cfgs
    - if too inefficient: only fetch the first config from each group
    """

    assert runs_to_merge is None, "Run merging logic not implemented"

    yaml_paths = {} # Dict[(str, str), Path]
    json_paths = defaultdict(dict) # Dict[(str, str), Dict[str, List[Path]]]

    for project in projects_dirs:
        project_path = Path(project)
        project_name = project_path.name

        for group in project_path.iterdir():
            if not group.is_dir():
                logger.info(f"{group} is not a run group")
                continue

            #print("Looking at ", group)

            try:
                # If there is no sub-runs, but all files are directly in the dir
                jp, yp = _process_run(group)

                if jp is not None:
                    yaml_paths[(project_name, group.name)] = yp
                    json_paths[(project_name, group.name)][""] = jp
            except ValueError:
                for run in group.iterdir():
                    if not run.is_dir():
                        logger.info(f"{group} is not a run")
                        continue

                    try:
                        jp, yp = _process_run(run)
                        if jp is not None:
                            assert not run.name in json_paths[(project_name, group.name)], f"{run.name} already present in {group.name}"
                            json_paths[(project_name, group.name)][run.name] = jp
                            if (project_name, group.name) not in yaml_paths:
                                yaml_paths[(project_name, group.name)] = yp

                    except ValueError:
                        logger.warning(f"{run} has no head_r0.yaml, skipping")

    json_paths_flat = []
    for (proj, group), runs in json_paths.items():
        for run, paths in runs.items():
            json_paths_flat.extend([
                {"proj": proj, "group": group, "run": run, "path": p} for p in paths
            ])

    yaml_paths_flat = [{"proj": proj, "group": group, "path": p} for (proj, group), p in yaml_paths.items()]

    with ThreadPool(n_threads) as pool:
        print(f"Loading {len(yaml_paths_flat)} yaml files")
        yaml_files = pool.map(_load_yaml, yaml_paths_flat)
        print(f"Loading {len(json_paths_flat)} json files")
        json_files = pool.map(_load_json, json_paths_flat)

    cfgs = {
        (e["proj"], e["group"]): {**e["data"], "cfg_path": str(e["path"])}
        for e in yaml_files
    }

    df_list = []

    for e in tqdm(json_files):
        group = e["group"]
        proj = e["proj"]
        run = e["run"]
        stem = e["path"].stem.removesuffix("_ava_results")
        stem = stem.replace("_r0", "")
        parts = stem.split("_")
        if len(parts) == 5:
            head, split, _, epoch, step = parts
        elif len(parts) == 4:
            head, split, _, epoch = parts
            step = "0"
        else:
            raise ValueError(f"Cannot parse file stem {stem}") 
        
        try:
            for k, v in e["data"].items():
                df_list.append({
                    "proj": proj,
                    "group": group,
                    "run": run.removeprefix(group),
                    "head": int(h) if (h := head.removeprefix("head")) else -1,
                    "epoch": float(epoch.removeprefix("epoch")),
                    "step": int(step.removeprefix("step")),
                    "split": split,
                    "metric": k,
                    "val": v,
                })
        except ValueError as e:
            print(f"{stem=}, {e=}")
            raise e

    df = pd.DataFrame(df_list)

    return df, cfgs

def _load_yaml(info):
    with info["path"].open() as fp:
        data = yaml.safe_load(fp)
    return {**info, "data": data}

def _load_json(info):
    with info["path"].open() as fp:
        data = json.load(fp)
    return {**info, "data": data}

def _process_run(group):

    children = list(group.iterdir())

    yaml_path = [c for c in children if c.name == "head_r0.yaml"]

    if not yaml_path:
        raise ValueError("No run")
    
    #print("  Found yaml file")

    assert len(yaml_path) == 1
    json_paths = list(group.glob("*.json"))

    if json_paths:
        #print("    Found json files")
        # Excluding runs where no json file was logged
        return json_paths, yaml_path[0]
    else:
        return None, None




def removesuffix(self, suffix):
    if self.endswith(suffix):
        return self[: -len(suffix)]
    return self


def removeprefix(self, prefix):
    if self.startswith(prefix):
        return self[len(prefix) :]
    return self

