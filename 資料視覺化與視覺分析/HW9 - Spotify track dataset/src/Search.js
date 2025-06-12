const input_g = d3.select("#input_g");
const g_sug = d3.select("#g_sug");

const input_a = d3.select("#input_a");
const a_sug = d3.select("#a_sug");

input_g.on("input", function () {
  SearchFor(this, g_sug, genreG, "g");
});

input_g.on("focus", function () {
  SearchFor(this, g_sug, genreG, "g");
});

input_a.on("input", function () {
  SearchFor(this, a_sug, artistG, "a");
});

input_a.on("focus", function () {
  SearchFor(this, a_sug, artistG, "a");
});

function SearchFor(input, sug, G, flag) {
  sug.html("");

  const inputRect = input.getBoundingClientRect();
  const inputTop = inputRect.bottom + window.scrollY;
  const inputLeft = inputRect.left + window.scrollX;

  sug.style("top", inputTop + "px").style("left", inputLeft + "px");

  const inputText = input.value.toLowerCase();

  const filteredData = G.filter((d) => d.toLowerCase().startsWith(inputText));

  if (filteredData.length > 0) {
    sug
      .style("display", "block")
      .selectAll(".suggestion-item")
      .data(filteredData.slice(0, 100))
      .enter()
      .append("div")
      .attr("class", "suggestion-item")
      .text((d) => d)
      .on("click", function () {
        // 將點擊的推薦選項填入搜尋框
        input.value = d3.select(this).text();
        // 清空推薦列表
        sug.html("");
        sug.style("display", "none");
      	if(flag=="g")
        {
        	bar_gen_flag = input.value;
      		bar_flag = "SOG"
      		d3.select("#specific-genre").property("checked", true);
        }
      	else if(flag=="a")
        {
        	bar_art_flag = input.value;
      		bar_flag = "SOA" 
      		d3.select("#specific-artist").property("checked", true);
        }
      	svgContainer.selectAll("*").remove();
        Leaderboard();
      });
  } else {
    sug.style("display", "none");
  }
}

document.addEventListener("click", function(event) {
  const isClickInsideInputOrSuggestion = input_g.node().contains(event.target) || g_sug.node().contains(event.target) ||
    input_a.node().contains(event.target) || a_sug.node().contains(event.target);

  if (!isClickInsideInputOrSuggestion) {
    // Click occurred outside the input and suggestion box, hide suggestion box
    g_sug.style("display", "none");
    a_sug.style("display", "none");
  }
});

// Existing code remains unchanged

// Rest of your code...
