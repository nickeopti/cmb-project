#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)

v_num <- args[1]
metrics_path <- paste(
    "logs",
    "lightning_logs",
    paste("version_", v_num, sep = ""),
    "metrics.csv",
    sep = "/"
)

metrics <- read_csv(metrics_path)

summarised_metrics <- metrics |>
    group_by(epoch) |>
    summarise(train_loss = mean(train_loss))

X11()
summarised_metrics |>
    ggplot() +
    aes(x = seq_len(nrow(summarised_metrics)), y = train_loss) +
    geom_line() +
    geom_smooth() +
    ylim(0, NA) +
    ylab("train loss") +
    xlab("epoch")
Sys.sleep(Inf)
