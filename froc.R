#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

data <- read_csv(args[1])

X11()
data |>
    group_by(method, threshold) |>
    summarise(
        tpr = sum(tp) / (sum(tp) + sum(fn)),
        mfp = mean(fp)
    ) |>
    View()
data |>
    group_by(method, threshold) |>
    summarise(
        tpr = sum(tp) / (sum(tp) + sum(fn)),
        mfp = mean(fp)
    ) |>
    ggplot() +
    aes(x = mfp, y = tpr, group = method, colour = method) +
    geom_point()

Sys.sleep(Inf)
