/*Sortable Bar*/
const RenderBarChart = (svg_container,data_, head, tail,x_att,y_att,w,h) =>{

  const color_keys = [
    'track_name','artists','track_genre'
  ];
  
  var color = d3
    .scaleOrdinal()
    .domain(color_keys)
    .range(d3.schemeDark2);
  
  const inner_margin = {top: 30, right: 30, bottom: 150, left: 80},
    inner_width = w - inner_margin.left - inner_margin.right,
    inner_height = h - inner_margin.top - inner_margin.bottom;
  
  
  var svg_ = svg_container 
    .append('svg')
    .attr("width", w)
    .attr("height", h)
    .append("g")
      .attr("transform", `translate(${inner_margin.left},${inner_margin.top})`);

      var data__ = data_.sort(function(b, a) {
        return a[y_att] - b[y_att];
      });  
      if(x_att==='track_name')
      {
        //只顯示head~tail的資料
        extendedData= data__.slice(head-1, tail*3);
    
        const uniqueData = Array.from(new Set(extendedData.map(item => item.track_name)))
          .map(track_name => extendedData.find(item => item.track_name === track_name));
    
        // Sort the unique data by 'popularity'
        data__ = uniqueData.sort((a, b) => b[y_att] - a[y_att]).slice(0, tail - head + 1); 
      }
      else
      {
        data__ =data__.slice(head-1, tail); 
      }
          
  const x = d3.scaleBand()
    .range([ 0, inner_width])
    .domain(data__.map(d => d[x_att]))
    .padding(0.2);
  
  svg_.append("g")
    .attr("transform", `translate(0, ${inner_height})`)
    .call(d3.axisBottom(x))
    .selectAll("text")
      .attr("transform", "translate(-10,0)rotate(-45)")
      .style("text-anchor", "end");
  
  svg_.append("text")
    .attr("transform", `translate(${inner_width / 2},${h-40})`)
    .style("text-anchor", "middle")
    .text(x_att)
    .style("font-family", "sans-serif");


  // Add Y axis
  const y = d3.scaleLinear()
    .domain([0, 100])
    .range([ inner_height,0]);
  svg_.append("g")
    .call(d3.axisLeft(y));
  
  svg_.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 20 - inner_margin.left)
    .attr("x", 0 - (inner_height ) / 2)
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .text(y_att)
    .style("font-family", "sans-serif");

  // Bars
  svg_.selectAll("mybar")
    .data(data__)
    .join("rect")
      .attr("x", d => x(d[x_att]))
      .attr("y", d => y(d[y_att]))
      .attr("width", x.bandwidth())
      .attr("height", d => inner_height - y(d[y_att]))
      .attr("fill", color(x_att))
      .each(function (d) {
        d3.select(this)
          .on("mouseover", function (event) {
            d3.select(this).attr("stroke", "red");
            const tooltipWidth = tooltip.node().offsetWidth;
            const tooltipX = Math.min(event.pageX, window.innerWidth - tooltipWidth-30);
            tooltip.transition()
              .duration(200)
              .style("opacity", .9);
          	if (x_att === 'track_name') {
              tooltip.html(`<div style="font-family: sans-serif;">
                            <strong>${d[x_att]}</strong><br>
                             Popularity: ${d[y_att]}<br>
                             Artist: ${d.artists}<br>
                             Album: ${d.album_name}<br>
                             Genre: ${d.track_genre}
                             </div>`)
            }
          	else{
            	tooltip.html(`<div style="font-family: sans-serif;">
                            <strong>${d[x_att]}</strong><br>
                             Popularity: ${d[y_att].toFixed(2)}
                             </div>`)
            }
            tooltip
              .style("left", (tooltipX) + "px")
              .style("top", (event.pageY - 28) + "px");
          })
          .on("mouseout", function () {
            d3.select(this).attr("stroke", "none");
            tooltip.transition()
              .duration(500)
              .style("opacity", 0);
          });
      });
  
  const tooltip = d3.select("body")
    .append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);
}
