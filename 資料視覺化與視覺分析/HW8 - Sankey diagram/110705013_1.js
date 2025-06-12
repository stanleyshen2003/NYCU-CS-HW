
var units = "Widgets";

// set the dimensions and margins of the graph
var margin = {top: 50, right: window.innerWidth/9.6, bottom: 0, left: window.innerWidth/9.6},
    width = window.innerWidth - margin.left - margin.right,
    height = 0.8* window.innerHeight - margin.top - margin.bottom;

// format variables
var formatNumber = d3.format(",.0f"),    // zero decimal places
    format = function(d) { return formatNumber(d) + " " + units; },
    color = d3.scaleOrdinal(d3.schemeCategory20);

// append the svg object to the body of the page
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", 
          "translate(" + margin.left + "," + margin.top + ")");

// Set the sankey diagram properties
var sankey = d3.sankey()
    .nodeWidth(36)
    .nodePadding(40)
    .size([width, height]);

var path = sankey.link();

const render = (data) => {
  svg.selectAll('*').remove();
  
  //set up graph in same style as original example but empty
  graph = {"nodes" : [], "links" : []};

  data.forEach(function (d) {
    graph.nodes.push({ "name": d.source });
    graph.nodes.push({ "name": d.target });
    graph.links.push({ "source": d.source,
                       "target": d.target,
                       "value": +d.value });
   });

  // return only the distinct / unique nodes
  graph.nodes = d3.keys(d3.nest()
    .key(function (d) { return d.name; })
    .object(graph.nodes));

  // loop through each link replacing the text with its index from node
  graph.links.forEach(function (d, i) {
    graph.links[i].source = graph.nodes.indexOf(graph.links[i].source);
    graph.links[i].target = graph.nodes.indexOf(graph.links[i].target);
  });

  // now loop through each nodes to make nodes an array of objects
  // rather than an array of strings
  graph.nodes.forEach(function (d, i) {
    graph.nodes[i] = { "name": d };
  });

  sankey
      .nodes(graph.nodes)
      .links(graph.links)
      .layout(32);

  // add in the links
  var link = svg.append("g").selectAll(".link")
      .data(graph.links)
    .enter().append("path")
      .attr("class", "link")
      .attr("d", path)
      .style("stroke-width", function(d) { return Math.max(1, d.dy); })
      .sort(function(a, b) { return b.dy - a.dy; });

  // add the link titles
  link.append("title")
        .text(function(d) {
    		return d.source.name + " → " + 
                d.target.name + "\n" + "amount: "+ d.value });

  // add in the nodes
  var node = svg.append("g").selectAll(".node")
      .data(graph.nodes)
    .enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { 
		  return "translate(" + d.x + "," + d.y + ")"; })
      .call(d3.drag()
        .subject(function(d) {
          return d;
        })
        .on("start", function() {
          this.parentNode.appendChild(this);
        })
        .on("drag", dragmove));

  // add the rectangles for the nodes
  node.append("rect")
      .attr("height", function(d) { return d.dy; })
      .attr("width", sankey.nodeWidth())
      .style("fill", function(d) { 
		  return d.color = color(d.name.replace(/ .*/, "")); })
      .style("stroke", function(d) { 
		  return d3.rgb(d.color).darker(2); })
    .append("title")
      .text(function(d) { 
		  return d.name + "\n" + format(d.value); });

  // add in the title for the nodes
  node.append("text")
      .attr("x", sankey.nodeWidth()/2)
      .attr("y", function(d) { return d.dy / 2; })
      .attr("dy", ".2em")
      .attr("text-anchor", "middle")
      .attr("transform", null)
  		.attr("text-align", "center")
  		.style("font-family", "sans-serif")
  		.style("font-weight", "bold")
  		.style("border","border: 1px solid transparent")
  		.style("text-shadow","none")
      .text(function(d) { 
    			var split = d.name.split('_');
    			return split[split.length-1]; });
    // .filter(function(d) { return d.x < width / 2; })
    //   .attr("x", sankey.nodeWidth()/2)
    //   .attr("text-anchor", "middle")
    // .attr("text-align", "center");

  // the function for moving the nodes
  function dragmove(d) {
    d3.select(this)
      .attr("transform", 
            "translate(" 
               + d.x + "," 
               + (d.y = Math.max(
                  0, Math.min(height - d.dy, d3.event.y))
                 ) + ")");
    sankey.relayout();
    link.attr("d", path);
  }
};

const columns = ["buying", "maintenance", "doors", "people", "luggage_boot", "safety"];
// load the data
var dataG;
d3.text("https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/HW8%20-%20Sankey%20diagram/sankey.data", function (text) {
  var data = d3.csvParseRows(text, (d, i) => {
    return {
      buying: d[0],
      maintenance: d[1],
      doors: d[2],
      people: d[3],
      luggage_boot: d[4],
      safety: d[5],
    };
  });

  var datacount = {};

  for (var i = 0; i < columns.length; i++) {
    datacount[columns[i]] = {};
    for (var j = 0; j < columns.length; j++) {
      datacount[columns[i]][columns[j]] = [];
    }
  }

  data.forEach((row) => {
    columns.forEach((c1) => {
      columns.forEach((c2) => {
        var d1 = c1 + "_" + row[c1];
        var d2 = c2 + "_" + row[c2];
        var matched = datacount[c1][c2].find(function (dict) {
          return dict.source === d1 && dict.target === d2;
        });

        if (matched) {
          matched.value += 1;
        } else {
          var emptydict = {};
          emptydict.source = d1;
          emptydict.target = d2;
          emptydict.value = 1;
          datacount[c1][c2].push(emptydict);
        }
      });
    });
  });

  
	dataG = datacount;
  
  data = [];
  for(var i=0;i<columns.length-1;i++){
    data = data.concat(dataG[columns[i]][columns[i+1]]);
    
  }
  render(data);
});


// add top botton
const buttonContainer = document.getElementById(
  'button-container'
);
buttonContainer.addEventListener(
  'dragstart',
  (e) => {
    e.dataTransfer.setData(
      'text/plain',
      e.target.id
    );
  }
);
buttonContainer.addEventListener(
  'dragover',
  (e) => {
    e.preventDefault();
  }
);
buttonContainer.addEventListener('drop', (e) => {
  e.preventDefault();
  const fromId = e.dataTransfer.getData(
    'text/plain'
  );
  const toId = e.target.id;

  if (fromId !== toId) {
    const fromButton = document.getElementById(
      fromId
    );
    const toButton = document.getElementById(
      toId
    );
    const fromRect = fromButton.getBoundingClientRect();
    const toRect = toButton.getBoundingClientRect();
    //buttonContainer.insertBefore(fromButton, toButton);
    if (fromRect.left < toRect.left) {
      if (toButton.nextSibling === null) {
        buttonContainer.appendChild(fromButton);
      } else {
        buttonContainer.insertBefore(
          fromButton,
          toButton.nextSibling
        );
      }
    } else {
      buttonContainer.insertBefore(
        fromButton,
        toButton
      );
    }
    printButtonSequence();
  }
});




function printButtonSequence() {
  const buttons = Array.from(
    buttonContainer.getElementsByClassName(
      'sortable-button'
    )
  );
  const sequence = buttons.map(
    (button) => button.id
  );
  console.log(sequence);
  data = [];
  for(var i=0;i<columns.length-1;i++){
    data = data.concat(dataG[sequence[i]][sequence[i+1]]);
    
  }
  render(data);
}