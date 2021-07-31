library(visNetwork)
library(igraph)
library(tidyverse)

el <- read_csv("test_el.csv")
author_inds <- read_csv("authors_df.csv")

el <- el %>% 
  left_join(author_inds, by = c("from" = "index")) %>% 
  select("from_id" = author_id, to)

el <- el %>% 
  left_join(author_inds, by = c("to" = "index")) %>% 
  select(from_id, "to_id" = author_id)

el <- el %>% 
  left_join(members %>% select(member_id, name), by = c("from_id" = "member_id"))

el <- el %>% 
  select("from_name" = name, to_id) %>% 
  left_join(members %>% select(member_id, name), by = c("to_id" = "member_id")) %>%
  select(from_name, "to_name"=name)

g <- graph_from_data_frame(el, directed = FALSE)
g <- g %>% igraph::simplify()

V(g)$size <- log(degree(g)) * 10

options(viewer=NULL)
visIgraph(g)

degree(g)
el %>% count(from_name) %>% left_join(el %>% count(to_name), by = c("from_name" = "to_name")) %>% mutate(sum = n.x + n.y) %>% View
