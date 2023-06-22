#!/usr/bin/env python3

import os
import sys

from snakemake.utils import read_job_properties
from snakemake.utils import format as sformat

jobscript = sys.argv[1]
job_properties = read_job_properties(jobscript)

qtime = job_properties["cluster"]["time"]
qnodes = job_properties["cluster"]["nodes"]
qcpus = job_properties["cluster"]["cpus"]
qmem = job_properties["cluster"]["mem"]
qname = job_properties["cluster"]["name"]

os.system(
    sformat(
        "qsub -lwalltime={qtime} -lselect={qnodes}:ncpus={qcpus}:mem={qmem} -N {qname} -j oe -V"
    )
)