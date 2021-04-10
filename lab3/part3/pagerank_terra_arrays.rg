import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c
sqrt = regentlib.sqrt(double)
fspace Page {
  rank : double,
  prevrank : double,
  numlinks : int,
  summation : double,
  summation_array : region(ispace(int1d, 64),double)
}

fspace Summation{
summation : double
}

struct PtrArray{
  data: &(&double);
  N: int
}

terra PtrArray:init(size: int)
   self.data = [&(&double)](c.malloc(size * sizeof(int32)))
   self.N = size
end


function Array(ElemType)  
 local struct ArrayType{
    data: &ElemType;
    N: int
 }
  -- Every struct has a namespace for associated methods.
  -- Similar to, but a bit more restrictive than, methods in object-oriented languages.
  terra ArrayType:get(i: int): ElemType
    return self.data[i]
  end

  terra ArrayType:init(size: int)
      self.data = [&ElemType](c.malloc(size * sizeof(ElemType)))
      self.N = size
  end
  
  return ArrayType
end

DoubleArray = Array(double)
--PtrArray = Array(&double)

--
-- TODO: Define fieldspace 'Link' which has two pointer fields,
--       one that points to the source and another to the destination.
--
fspace Link(r : region(Page)) {
  srcptr : ptr(Page, r),
  destptr : ptr(Page, r),
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
    reads (r_pages.rank,r_pages.prevrank) --, reads writes (r_pages.summation)
  do
    var sum = 0.0
    for page in r_pages do
    	--c.printf("sum %f \n",sum)
        sum += (page.rank - page.prevrank) * (page.rank - page.prevrank)
--	page.prevrank = page.rank
--	page.summation = 0.0
    end
    
    sum = sqrt(sum)
  return sum	
end

task final_ranks(r_pages : region(Page),
                 damp : double,		  
                 numpages : int,
		 parts : int,
                 sum_r : PtrArray
                 )
where
  reads(r_pages.summation_array), writes(r_pages.rank)
do
  var temp = 0.0
  for page in r_pages do
    var page_idx : int = page
    for j = 0, parts do
      temp += sum_r.data[page_idx][j] 
    end
   temp = temp * damp
   temp += (1-damp) / numpages
   page.rank = temp
  end
end

__demand(__leaf) 
task update_ranks(r_pages : region(Page),
     	          r_src : region(Page),
                  r_links : region(Link(wild)),
                  sum_r : PtrArray,
	          damp : double,
                  numpages : int,
		  summation_idx : int
	)
where
  reads(r_src.prevrank,r_src.numlinks,r_links) --, reads writes(sum_r)
do
  for link in r_links do
-- sum_calc (r_pages,r_src ,r_links )     	  
    var tmp_ptr = dynamic_cast(ptr(Page,r_pages),link.destptr)
    var tmp_src_ptr = dynamic_cast(ptr(Page,r_src),link.srcptr)
    var ptr_cnvrt : int = tmp_ptr	
    sum_r.data[summation_idx][ptr_cnvrt] += tmp_src_ptr.prevrank / tmp_src_ptr.numlinks	  
  end
--  for page in r_pages do
--    var temp = page.summation * damp
--    temp += (1-damp) / numpages
--    page.rank = temp
--  end            
--      final_rank(r_pages,damp,numpages)
      --c.printf("Rank_out = %f \n Page %d \n new_rank %f \n",page.rank,page,new_rank)    
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

  var p0 = partition(equal, r_links, c0)
  
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)
  var sums_r : PtrArray
  sums_r:init(config.parallelism)
  for i = 0, (config.parallelism) do
    var temp : DoubleArray
    temp:init(config.num_pages)
    for j = 0, (config.num_pages) do
      temp.data[j] = i + j
      c.printf("%lf \n", temp.data[j])
    end
    sums_r.data[i] = &(temp.data[0])
    c.printf("Pointer: %h", sums_r.data[i]) 
  end

  for i = 0, (config.parallelism) do
    for j = 0, (config.num_pages) do
      c.printf("%lf \n", sums_r.data[i][j])
    end
  end
  var dst_part = image(r_pages,p0,r_links.destptr)
  var src_part = image(r_pages,p0,r_links.srcptr) 
--  var temp = region(c0,region(r_pages.ispace,Summation))

  var num_iterations = 0
  var converged = false
  c.printf("Start \n")
   __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
 c.printf("Start \n")
  var ts_start = c.legion_get_current_time_in_micros()  
  while not converged do
    num_iterations += 1
--     __demand(__index_launch)
     for count in c0 do
     	 update_ranks(dst_part[count],src_part[count],p0[count],sums_r,config.damp,config.num_pages,count)
     end
--     __demand(__index_launch)
     for count in c0 do
     	 final_ranks(dst_part[count],config.damp,config.num_pages,config.parallelism,sums_r)
     end


   if num_iterations >= config.max_iterations then
      converged = true
      end
   if l2_norm(r_pages) < config.error_bound then
      converged = true
      end
   copy(r_pages.rank,r_pages.prevrank)
   fill(r_pages.summation,0.0)
   for i = 0, config.parallelism do
     for j = 0, config.num_pages do
       sums_r.data[i][j] = 0.0
     end
   end   

end
   __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)

