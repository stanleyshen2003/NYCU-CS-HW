(function () {
  'use strict';

  // set margin outside
  const marginWhole = {top: 10, right: 10, bottom: 10, left: 10},
      sizeWhole = window.innerHeight - marginWhole.left - marginWhole.right;

  // Create the svg area
  const svg = d3.select("#svg")
    .append("svg")
      .attr("width", sizeWhole  + marginWhole.left + marginWhole.right)
      .attr("height", sizeWhole  + marginWhole.top + marginWhole.bottom)
    .append("g")
      .attr("transform", `translate(${marginWhole.left},${marginWhole.top})`);
  var brushedPoints=[];

  // render function
  const render = (data,cols) =>{

    // set column names
    const allVar = cols;
    const numVar = allVar.length;

    // compute size of subgraph
    var mar = 20;
    var subsize = sizeWhole / numVar;

    // scale for subplot position
    const position = d3.scalePoint()
      .domain(allVar)
      .range([0, sizeWhole-subsize]);

    // color map
    const color = d3.scaleOrdinal()
      .domain(["Iris-setosa", "Iris-versicolor", "Iris-virginica" ])
      .range([ "#3CB7E4", "#FF8849", "#69BE28"]);

    for (var i in allVar){
      for (var j in allVar){

        // Get current variable name
        let var1 = allVar[i];
        let var2 = allVar[j];

        // draw histogram if column = row
        if (var1 === var2) {
          // x axis of histogram
          var extentX = d3.extent(data, function(d) { return d[var1] });
        	const x = d3.scaleLinear()
            .domain(extentX)
            .range([ 0, subsize-2*mar ]);

        // add g and move to sub-plot postion
        const tmp = svg
          .append('g')
          .attr("transform", `translate(${position(var1)+mar},${position(var2)+mar})`);

        // add x axis
        tmp.append("g")
          .attr("transform", `translate(0,${subsize-mar*2})`)
          .call(d3.axisBottom(x).ticks(3));

        // set the parameters for the histogram
        const histogram = d3.histogram()
             .value(function(d) { return d[var1]; })
         			// set max and min value for the graph
             .domain(x.domain())  
         			// ban ban * 15
             .thresholds(x.ticks(15));
          
        var data1 = data.filter(item => brushedPoints.includes(item) && item['class'] === 'Iris-setosa');
        var data2 = data.filter(item => brushedPoints.includes(item) && item['class'] === 'Iris-versicolor');
        var data3 = data.filter(item => brushedPoints.includes(item) && item['class'] === 'Iris-virginica');
  			// data 2 ban ban
        
        const bins = histogram(data);
        var bin1 = histogram(data1);
        var bin2 = histogram(data2);
        var bin3 = histogram(data3);
        // scale and draw y axis
        const y = d3.scaleLinear()
            .range([ 0,subsize-2*mar ])
            .domain([0, d3.max(bins, function(d) { return d.length; })]); 
  			
        const firstClassHeight = bin1.map(d => y(d.length));
        const secondClassHeight = bin2.map(d => y(d.length));
       // append the bar rectangles to the svg element
       tmp.append('g')
          .selectAll("rect")
          .data(bins) 
          .join("rect")
          .attr("x", 1)
          .attr("transform", d => {return `translate(${x(d.x0)},${subsize-2*mar-y(d.length)})`;})
         	.attr("width", function(d) { return x(d.x1) - x(d.x0)  ; })
          .attr("height", function(d) { return  y(d.length); })
          .style("fill", "#b8b8b8")
          .attr("stroke", "white");
        tmp.append('g')
          .selectAll("rect")
          .data(bin1)
          .enter()
          .append("rect")
          .attr("x", 1)
          .attr("transform", d => {return `translate(${x(d.x0)},${subsize-2*mar-y(d.length)})`;})
         	.attr("width", function(d) { return x(d.x1) - x(d.x0)  ; })
          .attr("height", function(d) { return  y(d.length); })
          .style("fill", "#3CB7E4")
          .attr("stroke", "white");
        tmp.append('g')
          .selectAll("rect")
          .data(bin2)
          .enter()
          .append("rect")
          .attr("x", 1)
          .attr("transform", (d,i) => {return `translate(${x(d.x0)},${subsize-2*mar-y(d.length)-firstClassHeight[i]})`;})
         	.attr("width", function(d) { return x(d.x1) - x(d.x0)  ; })
          .attr("height", function(d) { return  y(d.length); })
          .style("fill", "#FF8849")
          .attr("stroke", "white");
        tmp.append('g')
          .selectAll("rect")
          .data(bin3)
          .enter()
          .append("rect")
          .attr("x", 1)
          .attr("transform", (d,i) => {return `translate(${x(d.x0)},${subsize-2*mar-y(d.length)-firstClassHeight[i]-secondClassHeight[i]})`;})
         	.attr("width", function(d) { return x(d.x1) - x(d.x0)  ; })
          .attr("height", function(d) { return  y(d.length); })
          .style("fill", "#69BE28")
          .attr("stroke", "white");
        tmp.append("text")
        .attr("x", 1) // X coordinate of the text
        .attr("y", -5) // Y coordinate of the text
        .attr("fill", "black") // Text color
        .attr("font-size", "12px") // Font size
        .text(var1); // The text content
        continue;
        continue;
      }
  			/////////// scatter plot /////////////////
        // x scale
        var extentX = d3.extent(data, function(d) { return d[var1] });
        const x = d3.scaleLinear()
          .domain(extentX)
        	.nice()
          .range([ 0, subsize - 2*mar ]);

        // y scale
        var yextent = d3.extent(data, function(d) { return d[var2] });
        const y = d3.scaleLinear()
          .domain(yextent)
        	.nice()
          .range([ subsize - 2*mar, 0 ]);

        // set postion for sub-plot
        const tmp = svg
          .append('g')
          .attr("transform", `translate(${position(var1)+mar},${position(var2)+mar})`);

        // add tick
        tmp.append("g")
          .attr("transform", `translate(0,${subsize-mar*2})`)
          .call(d3.axisBottom(x).ticks(3));
        tmp.append("g")
          .call(d3.axisLeft(y).ticks(3));

        // add brush
        const brush = d3.brush()
            .extent([[0, 0], [subsize - 2 * mar, subsize - 2 * mar]])
            .on("end", brushended(data,var1,var2,x,y,cols));
        tmp.append("g")
          .call(brush);
        // draw circle
        const circles = tmp
          .selectAll("circle")
          .data(data)
          .join("circle")
            .attr("cx", function(d){ return x(d[var1]) })
            .attr("cy", function(d){ return y(d[var2]) })
            .attr("r", 1.75)
            .attr("fill", function(d) {
              // Check if the data point is in the brushedPoints array
              if (brushedPoints.includes(d)) {
                return color(d.class); // Use the original color for selected points
              } else {
                return "black"; // Set the fill color to black for unselected points
              }
            });

      }
    }
  };

  const columnsToInt = ['sepal length', 'sepal width', 'petal length', 'petal width'];
  d3.csv('https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/HW1%20-%20scattered%20plot/iris.csv').then(data => {
  	data.forEach(d => {
      columnsToInt.forEach(column => {
        d[column] = +d[column];
      });
    });
    data.pop();
    render(data,columnsToInt);
  });

  function brushended(data,var1,var2,x,y,cols) {
    return function(event) {
   	 const selection = event.selection;
      if (selection) {
        // get brushed points
        brushedPoints = data.filter(d => {
          const cx = x(d[var1]);
          const cy = y(d[var2]);
          return cx >= selection[0][0] && cx <= selection[1][0] && cy >= selection[0][1] && cy <= selection[1][1];
        });
        //console.log(brushedPoints);
        svg.selectAll('*').remove();
        render(data,cols);
      }
    }
  }

}());