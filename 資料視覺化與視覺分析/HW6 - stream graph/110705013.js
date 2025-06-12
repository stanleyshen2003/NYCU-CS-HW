(function () {
  'use strict';

  var dataG = {};
  var timeParser = d3.timeParse('%d/%m/%Y');
  var timeFormat = d3.timeFormat('%Y/%m/%d');
  const fixed_keys = [
    'house2',
    'house3',
    'house4',
    'house5',
    'unit1',
    'unit2',
    'unit3',
  ];
  var margin = {
      top: window.innerHeight / 10,
      right: window.innerWidth / 5,
      bottom: window.innerHeight / 10,
      left: window.innerWidth / 15,
    },
    width =
      window.innerWidth -
      margin.left -
      margin.right,
    height =
      window.innerHeight -
      margin.top -
      margin.bottom;

  // append the svg object to the body of the page
  var svg = d3
    .select('#svg')
    .append('svg')
    .attr(
      'width',
      width + margin.left + margin.right
    )
    .attr(
      'height',
      height + margin.top + margin.bottom
    )
    .append('g')
    .attr(
      'transform',
      'translate(' +
        margin.left +
        ',' +
        margin.top +
        ')'
    );
  function render(dimensions) {
    svg.selectAll('*').remove();

    // Add X axis
    var x = d3
      .scaleLinear()
      .domain(
        d3.extent(dataG, function (d) {
          return d.saledate;
        })
      )
      .range([0, width]);

    var results = [];
    for (let i = 3; i < dataG.length; i += 4) {
      results.push(dataG[i].saledate);
    }
    svg
      .append('g')
      .attr(
        'transform',
        'translate(0,' + height * 0.1 + ')'
      )
      .call(
        d3
          .axisBottom(x)
          .tickSize(height * 0.8)
          .tickValues(results)
          .tickFormat(timeFormat)
      )
      .select('.domain')
      .remove();

    // Customization
    svg
      .selectAll('.tick line')
      .attr('stroke', '#b8b8b8');

    // // Add X axis label:
    // svg.append("text")
    //     .attr("text-anchor", "end")
    //     .attr("x", width)
    //     .attr("y", height-30 )
    //     .text("Time (9/30)");

    const columnSums = new Array(dataG.length).fill(
      0
    );
    for (let i = 0; i < dataG.length; i++) {
      for (const elements of fixed_keys) {
        columnSums[i] += dataG[i][elements];
      }
    }

    // Add Y axis
    var y = d3
      .scaleLinear()
      .domain([-4000000, 4000000])
      .range([0, height]);

    // color palette
    var color = d3
      .scaleOrdinal()
      .domain(fixed_keys)
      .range(d3.schemeDark2);
    for (let i = 0; i < 7; i++) {
      console.log(d3.schemeDark2[i]);
    }
    //stack the data?
    var stackedData = d3
      .stack()
      .offset(d3.stackOffsetSilhouette)
      .keys(dimensions)(dataG);

    // create a tooltip
    var Tooltip = svg
      .append('text')
      .attr('x', 0)
      .attr('y', 0)
      .style('opacity', 0)
      .style('font-size', 17);

    // Three function that change the tooltip when user hover / move / leave a cell
    var mouseover = function (d) {
      Tooltip.style('opacity', 1);
      d3.selectAll('.myArea').style('opacity', 0.2);
      d3.select(this)
        .style('stroke', 'black')
        .style('opacity', 1);
    };
    var mousemove = function (d, i) {
      var grp = fixed_keys[i];
      Tooltip.text(grp).attr('x',width/2).style('font-weight', 'bold').style('text-anchor','middle');
    };
    var mouseleave = function (d) {
      Tooltip.style('opacity', 0);
      d3.selectAll('.myArea')
        .style('opacity', 1)
        .style('stroke', 'none');
    };

    // Area generator
    var area = d3
      .area()
      .x(function (d) {
        return x(d.data.saledate);
      })
      .y0(function (d) {
        return y(d[0]);
      })
      .y1(function (d) {
        return y(d[1]);
      });

    // Show the areas
    svg
      .selectAll('mylayers')
      .data(stackedData)
      .enter()
      .append('path')
      .attr('class', 'myArea')
      .style('fill', function (d) {
        return color(d.key);
      })
      .attr('d', area)
      .on('mouseover', mouseover)
      .on('mousemove', mousemove)
      .on('mouseleave', mouseleave);
  }
  //render(['sepal width', 'sepal length',  'petal width', 'petal length' ]);

  d3.csv('https://raw.githubusercontent.com/stanleyshen2003/Data-Visualization/main/HW6%20-%20stream%20graph/ma_lga_12345.csv').then(function (data) {
    // Create an empty object to store the transformed data
    var transformedData = [];
    var uniqueDates = new Set();
    var uniqueEle = new Set();
    data.forEach(function (d) {
      uniqueDates.add(d.saledate); // Assuming "date" is the date field in your data
      uniqueEle.add(d.type + d.bedrooms);
    });

    uniqueDates.forEach(function (saledate) {
      var entry = { saledate: saledate };

      uniqueEle.forEach(function (element) {
        var filtered = data.filter(function (d) {
          return (
            d.type + d.bedrooms == element &&
            d.saledate === saledate
          );
        });

        if (filtered.length > 0) {
          entry[element] = +filtered[0].MA;
        } else {
          entry[element] = 0;
        }
      });

      transformedData.push(entry);
    });

    transformedData.forEach(function (d) {
      d.saledate = timeParser(d.saledate);
    });

    transformedData.sort(function (a, b) {
      return d3.ascending(a.saledate, b.saledate);
    });

    dataG = transformedData;
    console.log(dataG);
    render([
      'house2',
      'house3',
      'house4',
      'house5',
      'unit1',
      'unit2',
      'unit3',
    ]);
  });

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
    console.log(fromId);
    console.log(toId);
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
      if (fromRect.top < toRect.top) {
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
    render(sequence);
  }

}());

