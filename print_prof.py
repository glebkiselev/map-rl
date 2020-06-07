import pstats

p = pstats.Stats('main.prof')

p.sort_stats('time').print_stats()