library(plotly)
library(ggplot2)

nsday <- data.frame(
  Simd = factor(c("AVX2","NSIMD","AVX2","NSIMD")),
  pme_rank = factor(c("0","0","-1","-1"), levels=c("-1","0")),
  perf = c(2.537, 2.607, 2.530, 2.498)
)

gcycle <- data.frame(
  Simd = factor(c("AVX2","NSIMD","AVX2","NSIMD")),
  pme_rank = factor(c("0","0","-1","-1"), levels=c("-1","0")),
  perf = c(2285.657, 2224.498, 2291.430,2339.114)
)


# Bar graph
p1 <- ggplot(data=nsday, aes(x=pme_rank, y=perf, fill=Simd)) +
  geom_bar(stat="identity", position=position_dodge()) + xlab("PME Rank") + ylab("Performance (ns/day)")
p1

p2 <- ggplot(data=gcycle, aes(x=pme_rank, y=perf, fill=Simd)) +
  geom_bar(stat="identity", position=position_dodge()) + xlab("PME Rank") + ylab("Performance (Gcycles)")
p2
  
