import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c
sqrt = regentlib.sqrt(double)
fspace Page {
  rank : double,
  prevrank : double,
  numlinks : int
}

--
-- TODO: Define fieldspace 'Link' which has two pointer fields,
--       one that points to the source and another to the destination.
--
fspace Link(r : region(Page)) {
  srcptr : ptr(Page, r),
  destptr : ptr(Page, r)
}

terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_ids(f : &c.FILE, page_ids : &uint32)
  return c.fscanf(f, "%d %d\n", &page_ids[0], &page_ids[1]) == 2
end

task initialize_graph(r_pages   : region(Page),
                      r_links   : region(Link(wild)),
                      damp      : double,
                      num_pages : uint64,
                      filename  : int8[512])
where
  reads writes(r_pages, r_links)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for page in r_pages do
    page.rank = 1.0 / num_pages
    page.prevrank = 1.0 / num_pages
    page.numlinks = 0.0
  end

  var f = c.fopen(filename, "rb")
  skip_header(f)
  var page_ids : uint32[2]
  for link in r_links do
    regentlib.assert(read_ids(f, page_ids), "Less data that it should be")
    var src_page = dynamic_cast(ptr(Page, r_pages), page_ids[0])
    var dst_page = dynamic_cast(ptr(Page, r_pages), page_ids[1])
    src_page.numlinks += 1
    --c.printf("Numlinks = %d \n" , src_page.numlinks)
    link.srcptr = src_page
    link.destptr = dst_page
  end
  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end

--
-- TODO: Implement PageRank. You can use as many tasks as you want.
--

task l2_norm(r_pages : region(Page)) : double
  where 
    reads (r_pages)
  do
    var sum = 0.0
    for page in r_pages do
    	--c.printf("sum %f \n",sum)
        sum += (page.rank - page.prevrank) * (page.rank - page.prevrank)
    end
    
    sum = sqrt(sum)
  return sum	
end


task update_ranks(r_pages : region(Page),
                r_links : region(Link(wild)),
                damp : double,
                numpages : int)
  where
    reads writes(r_pages, r_links)
  do
    for page in r_pages do
      var new_rank = 0.0
      for link in r_links do
        if link.destptr == &r_pages[page] then
	   var tmp_ptr = dynamic_cast(ptr(Page,r_pages),link.srcptr)
	   --new_rank += 1
           new_rank += tmp_ptr.prevrank / tmp_ptr.numlinks
	   --c.printf("%f  = %f  %d \n", new_rank , tmp_ptr.prevrank, tmp_ptr.numlinks);
	   --c.printf("%d LINKS \n",tmp_ptr.numlinks)
        end
	--c.printf("PPP %d %d\n" , link.destptr,page)
      end
      new_rank *= damp
      new_rank += (1-damp) / numpages
      page.rank = new_rank
      --c.printf("Rank_out = %f \n Page %d \n new_rank %f \n",page.rank,page,new_rank)
    end
    
end

task update_prev_rank(r_pages : region(Page))
  where
    reads writes(r_pages)
  do
    for page in r_pages do
      page.prevrank = page.rank
    end
  
end

task dump_ranks(r_pages  : region(Page),
                filename : int8[512])
where
  reads(r_pages.rank)
do
  var f = c.fopen(filename, "w")
  for page in r_pages do c.fprintf(f, "%g\n", page.rank) end
  c.fclose(f)
end

task toplevel()
  var config : PageRankConfig
  config:initialize_from_command()
  c.printf("**********************************\n")
  c.printf("* PageRank                       *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Pages  : %11lu *\n",  config.num_pages)
  c.printf("* Number of Links  : %11lu *\n",  config.num_links)
  c.printf("* Damping Factor   : %11.4f *\n", config.damp)
  c.printf("* Error Bound      : %11g *\n",   config.error_bound)
  c.printf("* Max # Iterations : %11u *\n",   config.max_iterations)
  c.printf("**********************************\n")

  -- Create a region of pages
  var r_pages = region(ispace(ptr, config.num_pages), Page)
  --
  -- TODO: Create a region of links.
  --       It is your choice how you allocate the elements in this region.
  --
  var r_links = region(ispace(ptr, config.num_links), Link(wild))

  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)

  var num_iterations = 0
  var converged = false
  var ts_start = c.legion_get_current_time_in_micros()
  while not converged do
    num_iterations += 1

    update_ranks(r_pages, r_links, config.damp, config.num_pages)
    
    if num_iterations > config.max_iterations then
      converged = true
    end

    if l2_norm(r_pages) < config.error_bound then
      converged = true
    end
    update_prev_rank(r_pages)
  end
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)

