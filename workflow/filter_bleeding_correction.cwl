#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool
arguments:
  - docker
  - run
  - "-v"
  - $(runtime.outdir):/data
  - "-v"
  - $(inputs.raw_data.path):/input
  - $(inputs.docker_image)
  - filter_bleed
  - "--count-mat"
  - /input
  - "--data-dir"
  - /data
  - "--n-gene"
  - $(inputs.n_gene)
  - "--filter-type"
  - $(inputs.filter_type)
inputs:
  docker_image: string
  raw_data: File
  filter_type: string
  n_gene: string
outputs: []