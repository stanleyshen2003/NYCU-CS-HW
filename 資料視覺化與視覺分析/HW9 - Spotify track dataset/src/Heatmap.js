d3.select("body").append("div").attr("class", "tip").style("display", "none");
var corrG = 0;
var setcorr = (data, cols) => {
  corrG = jz.arr.correlationMatrix(data,cols)

}
const RenderHeatmap = (data, svg,w,h) => {
  var svg_ = svg.append('svg').attr('width', w).attr('height',h).attr("id", "Svg");
  if (corrG == 0){
    corrG = jz.arr.correlationMatrix(data, cols);
  }
  var extent = d3.extent(corrG.map(function(d) { return d.correlation; }).filter(function(d) { return d !== 1; }));
  var grid = data2grid.grid(corrG);
  var rows = cols.length;
  
  var inner_margin = { top: 40, bottom: 30, left: 100, right: 0 };
  var dim = d3.min([w,h])
  var inner_width = dim *0.7,
   	 inner_height = dim * 0.7;

  svg_ = svg_
    .attr("width", w)
    .attr("height", dim)
    .append("g")
    .attr("transform", "translate(" + (w-inner_width)/2+inner_margin.left + ", " + inner_margin.top + ")");

  var padding = .1;
  var x = d3.scaleBand()
    .range([0, inner_width])
    .paddingInner(padding)
    .domain(d3.range(1, rows + 1));
  
  var y = d3.scaleBand()
    .range([0, inner_height])
    .paddingInner(padding)
    .domain(d3.range(1, rows + 1));
  
  var x_axis = d3.axisBottom(x).tickFormat(function(d, i) { return cols[i]; });
  var y_axis = d3.axisLeft(y).tickFormat(function(d, i) { return cols[i]; });

  var axisXcreated = svg_.append("g")
    .attr("class", "x axis")
    .call(x_axis)
    .attr("transform", "translate(0, " + inner_height + ")");
  axisXcreated.selectAll('text')
    .style('text-anchor', 'end')
    .attr('dx', '-0.5em')
    .attr('dy', '0.5em')
    .attr('transform', 'rotate(-45)'); ;
  svg_.append("g")
    .attr("class", "y axis")
    .call(y_axis);

  
  var colorscale = chroma.scale(["tomato", "white", "steelblue"])
    .domain([-1, 0, extent[1]]);
  
  svg_.selectAll("rect")
  .data(grid, function(d) { return d.column_a + d.column_b; })
  .enter().append("rect")
  .attr("x", function(d) { return x(d.column); })
  .attr("y", function(d) { return y(d.row); })
  .attr("width", x.bandwidth())
  .attr("height", y.bandwidth())
  .style("fill", function(d) { return colorscale(d.correlation); })
  .on("mouseover", function(d) {
    d3.select(this).classed("selected", true);
		d_ = d3.select(this).datum()

    var [mouseX, mouseY] = d3.pointer(event);
    mouseX += (w - inner_width) / 2+20;
    mouseY += height * 0.2;

    const tooltipWidth = tooltip.node().offsetWidth;
    const tooltipX = Math.min(event.pageX, window.innerWidth - tooltipWidth-30);
    tooltip.transition()
        .duration(200)
        .style("opacity", .9);
    tooltip.html(`<div style="font-family: sans-serif;">
                            ${d_.column_x}, ${d_.column_y}<br>
                             Correlation: ${ d_.correlation.toFixed(2)}
                             </div>`)
    tooltip
        .style("left", mouseX + "px")
        .style("top", mouseY + "px");

    svg_.select(".x.axis .tick:nth-of-type(" + d_.column + ") text").classed("selected", true);
    svg_.select(".y.axis .tick:nth-of-type(" + d_.row + ") text").classed("selected", true);
    svg_.select(".x.axis .tick:nth-of-type(" + d_.column + ") line").classed("selected", true);
    svg_.select(".y.axis .tick:nth-of-type(" + d_.row + ") line").classed("selected", true);
  })
  .on("mouseout", function() {
    svg_.selectAll("rect").classed("selected", false);
    tooltip.transition()
              .duration(500)
              .style("opacity", 0);
    svg_.selectAll(".axis .tick text").classed("selected", false);
    svg_.selectAll(".axis .tick line").classed("selected", false);
  });
  
  const tooltip = d3.select("body")
    .append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

};

var cols = ['popularity','explicit','acousticness','danceability','energy','instrumentalness','liveness',
      'loudness','speechiness','valence','tempo','mode','key','duration_ms'];


const HeatLegend = (svg,w,h) => {
  var svg_ = svg.append('svg').attr('width',w).attr('height',h).attr("id", "Lgd");
  var dim = w;
  console.log(dim)
  var inner_margin = { top: 30, bottom: 0, left: dim * 0.33, right: dim*0.35 };

  var inner_width = w - inner_margin["left"] - inner_margin["right"] ,
    inner_height = dim;

  var legend_top = 50;
  var legend_height = 20;

  var legend_svg = svg_
    .append("svg")
    .attr("width", dim)
    .attr("height", legend_height + legend_top)
    .append("g")
    .attr("transform", "translate(" + inner_margin.left + ", " + legend_top + ")");

  var defs = legend_svg.append("defs");

  var gradient = defs.append("linearGradient")
    .attr("id", "linear-gradient");

  var stops = [
    { offset: 0, color: "tomato", value: -1 },
    { offset: .5, color: "white", value: 0 },
    { offset: 1, color: "steelblue", value: 1 }
  ];

  gradient.selectAll("stop")
    .data(stops)
    .enter().append("stop")
    .attr("offset", function(d) { return (100 * d.offset) + "%"; })
    .attr("stop-color", function(d) { return d.color; });

  legend_svg.append("rect")
    .attr("width", inner_width)
    .attr("height", legend_height)
    .style("fill", "url(#linear-gradient)");

  legend_svg.selectAll("text")
    .data(stops)
    .enter().append("text")
    .attr("x", function(d) { return inner_width * d.offset; })
    .attr("dy", -3)
    .style("text-anchor", function(d, i) { return i == 0 ? "start" : i == 1 ? "middle" : "end"; })
    .style("font-family", "sans-serif")
    .text(function(d, i) { return d.value.toFixed(2); })
}