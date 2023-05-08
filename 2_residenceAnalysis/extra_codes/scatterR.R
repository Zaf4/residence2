library(tidyverse)



durations <- read_csv('./data/durations_minimized.csv')

durations |> 
  melt(id=names)


ax<-ggplot(data=durations)+
  geom_point(aes())
