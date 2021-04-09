import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c
sqrt = regentlib.sqrt(double)
fspace Page {
  rank : double,
  prevrank : double,
  numlinks : int,
  summation : double

}

--
-- TODO: Define fieldspace 'Link' which has two pointer fields,
--       one that points to the source and another to the destination.
--
fspace Link(r : region(Page)) {
  srcptr : ptr(Page, r),
  destptr : ptr(Page, r),
  cont : double
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
    page.summation = 0.0
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

--__demand(__parallel)
task update_cont(r_source: region(Page),
     		r_links : region(Link(wild))
     			   )
	where
	    reads (r_source) ,
	    reads writes(r_links)
	do
	    for link in r_links do
	    var tmp_ptr = dynamic_cast(ptr(Page,r_source),link.srcptr)
	    link.cont = tmp_ptr.prevrank / tmp_ptr.numlinks	
	    end
end

--__demand(__parallel)
task update_sum(r_pages : region(Page),
     		damp : double,
                numpages : int
		)
where
	reads writes(r_pages)
do
      for page in r_pages do
      page.summation *= damp
      page.summation += (1-damp) / numpages
      page.rank = page.summation
      end

end

--__demand(__parallel)
task update_ranks(r_pages : region(Page),
                r_links : region(Link(wild))                
	)
  where
    reads writes(r_pages, r_links)
  do
      for link in r_links do
     	   var tmp_ptr = dynamic_cast(ptr(Page,r_pages),link.destptr)
           tmp_ptr.summation += link.cont	  
      end
      --c.printf("Rank_out = %f \n Page %d \n new_rank %f \n",page.rank,page,new_rank)    
end

task update_prev_rank(r_pages : region(Page))
  where
    reads writes(r_pages)
  do
    for page in r_pages do
      page.prevrank = page.rank
      page.summation = 0.0
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
  c.printf("* # Parallel Tasks : %11u *\n",   config.parallelism)
  
  c.printf("**********************************\n")

  -- Create a region of pages
  var r_pages = region(ispace(ptr, config.num_pages), Page)
  --
  -- TODO: Create a region of links.
  --       It is your choice how you allocate the elements in this region.
  --
  var r_links = region(ispace(ptr, config.num_links), Link(wild))

  --   
  -- TODO: Create partitions for links and pages.
  --       You can use as many partitions as you want.
  --
  var c0 = ispace(int1d, config.parallelism)
  var image0 = partition(equal,r_links, c0)
--  var image0 = preimage(r_links,p0,r_links.destptr)
--  var srcimage = image(r_pages,image0,r_links.srcptr)  
  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)
  var srcimage = image(r_pages,image0,r_links.srcptr)
  var p0 = partition(equal,r_pages,c0)  
  var dest_edge = preimage(r_links,p0,r_links.destptr)

  var num_iterations = 0
  var converged = false
  c.printf("Start \n")
   __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
 c.printf("Start \n")
  var ts_start = c.legion_get_current_time_in_micros()  
  while not converged do
    num_iterations += 1

    --update_ranks(r_pages, r_links, config.damp, config.num_pages)-    
__demand(__index_launch)
    for count in c0 do    
    	update_cont(srcimage[count],image0[count])
    end
    __demand(__index_launch)
     for count in c0 do
	update_ranks(p0[count],dest_edge[count])
     end
    __demand(__index_launch)
    for count in c0 do
        update_sum(p0[count],config.damp,config.num_pages)
      end	
--  for count in c0 do
--	for page in p0[count] do
--	    c.printf("Page:%d \n",page)
--	end
--	for link in image0[count] do
--	c.printf("Link %d, dest %d source %d\n",link,link.destptr,link.srcptr)
--	end
--	for page in srcimage[count] do
--	c.printf("Partition %d source %d \n" , count,page)
--	end
--  end
    if num_iterations >= config.max_iterations then
      converged = true
    end
--    c.printf("n")
    if l2_norm(r_pages) < config.error_bound then
      	 converged = true
   end
--__demand(__index_launch)
 for count in c0 do
    update_prev_rank(p0[count])
end
--    break
  end
   __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)