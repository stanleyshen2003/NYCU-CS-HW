(function () {
  'use strict';

  d3.select("body").append("div").attr("class", "tip").style("display", "none");

  var cols = [ 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'];
  const render = (data,svg,title)=>{
  // create correlation matrix
  var corr = jz.arr.correlationMatrix(data, cols); 
  // map the extent of correlation matrix
  var extent = d3.extent(corr.map(function(d){ return d.correlation; }).filter(function(d){ return d !== 1; }));
  // append element #grid
  var grid = data2grid.grid(corr);
  // get number of rows
  var rows = cols.length;
  // set margin
  var margin = {top: 30, bottom: 30, left: 100, right: 0};
  // set dimension for d3
  var dim = d3.min([window.innerWidth * .6, window.innerHeight * .6]);
  // get width & height of svg
  var width = dim *0.5, 
      height = dim *0.5;

  // create svg_
  svg = svg
    	.attr("width", dim)
      .attr("height", dim)
      .append("g")
      .attr("transform", "translate(" + margin.left + ", " + margin.top + ")");
  svg.append('text').text(title) 
    .attr('class','title')
    .attr("transform", "translate(" + 0 + ", " + -10 + ")");
  // set padding
  var padding = .1;
  // set scalor for tick
  var x = d3.scaleBand()
    .range([0, width])
    .paddingInner(padding)
    .domain(d3.range(1, rows + 1));
  var y = d3.scaleBand()
    .range([0, height])
    .paddingInner(padding)
    .domain(d3.range(1, rows + 1));
  // create tick
  var x_axis = d3.axisBottom(x).tickFormat(function(d, i){ return cols[i]; });
  var y_axis = d3.axisLeft(y).tickFormat(function(d, i){ return cols[i]; });

  // call the ticks
  var axisXcreated = svg.append("g")
      .attr("class", "x axis")
      .call(x_axis)
    	.attr("transform", "translate(0, " + height + ")");
  axisXcreated.selectAll('text')
    .style('text-anchor', 'end')
    .attr('dx', '-0.5em')
    .attr('dy', '0.5em')
    .attr('transform', 'rotate(-45)');svg.append("g")
      .attr("class", "y axis")
      .call(y_axis);

  // create scaler for color
  var colorscale = chroma.scale(["tomato", "white", "steelblue"])
    .domain([-1, 0, extent[1]]);
  // draw the rectengles
  svg.selectAll("rect")
      .data(grid, function(d){ return d.column_a + d.column_b; }) // function: concat columnNames to form a 
    	.enter().append("rect")
      .attr("x", function(d){ return x(d.column); })
      .attr("y", function(d){ return y(d.row); })
      .attr("width", x.bandwidth())
      .attr("height", y.bandwidth())
      .style("fill", function(d){ return colorscale(d.correlation); })
    	.transition()
      .style("opacity", 10);


  svg.selectAll("rect")
    	.on("mouseover", function(d){
          // add event listher to change class selected to true
          d3.select(this).classed("selected", true);

          d3.select(".tip")
              .style("display", "block")
              .html(d.column_x + ", " + d.column_y + ": " + d.correlation.toFixed(2));

          var row_pos = y(d.row);
          var col_pos = x(d.column);
          var tip_pos = d3.select(".tip").node().getBoundingClientRect();
          var tip_width = tip_pos.width;
          var tip_height = tip_pos.height;
          var grid_pos = svg.node().getBoundingClientRect();
          var grid_left = grid_pos.left;
          var grid_top = grid_pos.top;
          var left = grid_left + col_pos + margin.left - 20;
          var top = grid_top + row_pos + margin.top - tip_height - 5;
  				//console.log(row_s);
          d3.select(".tip")
              .style("left", left + "px")
              .style("top", top + "px");

          svg.select(".x.axis .tick:nth-of-type(" + d.column + ") text").classed("selected", true);
          svg.select(".y.axis .tick:nth-of-type(" + d.row + ") text").classed("selected", true);
          svg.select(".x.axis .tick:nth-of-type(" + d.column + ") line").classed("selected", true);
          svg.select(".y.axis .tick:nth-of-type(" + d.row + ") line").classed("selected", true);

      })
    	.on("mouseout", function(){
          svg.selectAll("rect").classed("selected", false);
          d3.select(".tip").style("display", "none");
          svg.selectAll(".axis .tick text").classed("selected", false);
          svg.selectAll(".axis .tick line").classed("selected", false);
    	});
  };
  const Legend = () =>{
  // set dimension for d3
  	var dim = window.innerWidth;
    var margin = {top: 0, bottom: 0, left:dim * 0.1 , right: 30};
    
  // get width & height of svg
  var width = dim * 0.8; 
   // legend scale
      var legend_top = 80;
      var legend_height = 20;

      var legend_svg = d3.select(".Legend")
      		.append("svg")
          .attr("width", dim)
          .attr("height", legend_height + legend_top)
        	.append("g")
          .attr("transform", "translate("+margin.left+", " + legend_top + ")");

      var defs = legend_svg.append("defs");

      var gradient = defs.append("linearGradient")
          .attr("id", "linear-gradient");

      var stops = [{offset: 0, color: "tomato", value: -1}, {offset: .5, color: "white", value: 0}, {offset: 1, color: "steelblue", value: 1}];

      gradient.selectAll("stop")
          .data(stops)
        	.enter().append("stop")
          .attr("offset", function(d){ return (100 * d.offset) + "%"; })
          .attr("stop-color", function(d){ return d.color; });

      legend_svg.append("rect")
          .attr("width", width)
          .attr("height", legend_height)
          .style("fill", "url(#linear-gradient)");

      legend_svg.selectAll("text")
          .data(stops)
        	.enter().append("text")
          .attr("x", function(d){ return width * d.offset; })
          .attr("dy", -3)
          .style("text-anchor", function(d, i){ return i == 0 ? "start" : i == 1 ? "middle" : "end"; })
          .text(function(d, i){ return d.value.toFixed(2) ; });
  };


  d3.text("https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/HW3%20-%20correlation%20matrix/abalone.data", function(text) {
  var data = d3.csvParseRows(text, (d, i) => {
    return {
    sex: d[0],
    length: +d[1],
    diameter: +d[2],
    height: +d[3],
    whole_weight: +d[4],
    shucked_weight: +d[5],
    viscera_weight: +d[6],
    shell_weight: +d[7],
    rings: +d[8]
        };
  });
  var dataM = data.filter(function(d) {return d['sex'] === 'M';});
  var dataF = data.filter(function(d) {return d['sex'] === 'F';});
  var dataI = data.filter(function(d) {return d['sex'] === 'I';});
  var svgM = d3.select(".svgM");
  var svgF = d3.select(".svgF");
  var svgI = d3.select(".svgI");
  render(dataM,svgM,'Male');
  render(dataF,svgF,'Female');
  render(dataI,svgI,'Infant');
  Legend();
  });

}());
