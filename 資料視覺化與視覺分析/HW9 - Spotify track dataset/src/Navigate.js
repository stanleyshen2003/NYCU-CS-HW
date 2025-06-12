/* Navigtion Bar */

// Leaderboard
const L_Btn = d3.select("#LBtn");
const L_Div = d3.select("#L")

// Rader
const A_Btn = d3.select("#ABtn");
const A_Div = d3.select("#A")

// Correlation
const H_Btn = d3.select("#HBtn");


let Gener_;

// Click btn show setting
L_Btn.on("click", function () {
  L_Div.style('display', 'block');
  A_Div.style('display', 'none');
  A_Btn.style("background-color", "lightgrey");
  H_Btn.style("background-color", "lightgrey");
  L_Btn.style("background-color", "lightsteelblue");
  svgContainer.selectAll("*").remove();
  Leaderboard();
});

A_Btn.on("click", function () {
  L_Div.style('display', 'none');
  A_Div.style('display', 'block'); 
  L_Btn.style("background-color", "lightgrey");
  H_Btn.style("background-color", "lightgrey");
  A_Btn.style("background-color", "lightsteelblue");
  svgContainer.selectAll("*").remove();
  AnalysisAll();
});

H_Btn.on("click", function () {
  L_Div.style('display', 'none');
  A_Div.style('display', 'none'); 
  L_Btn.style("background-color", "lightgrey");
  A_Btn.style("background-color", "lightgrey");
  H_Btn.style("background-color", "lightsteelblue");
  svgContainer.selectAll("*").remove();
  CorrelationMatrix();
});


var L_option = d3.selectAll('input[name="L_option"]');

// Listen for changes in the radio group
L_option.on('change', function () {
    // Get the selected value
    var selectedValue = d3.select('input[name="L_option"]:checked').node().value;

    // Set bar_flag based on the selected value
    switch (selectedValue) {
        case 'song':
            setBarFlag('S');
            break;
        case 'artist':
            setBarFlag('A');
            break;
        case 'genre':
            setBarFlag('G');
            break;
        case 'specific-genre':
            setBarFlag('SOG');
            break;
        case 'specific-artist':
            setBarFlag('SOA');
            break;
        default:
            break;
    }
  	svgContainer.selectAll("*").remove();
  	Leaderboard();
});

// Function to set the bar_flag
function setBarFlag(value) {
    // Assuming bar_flag is a global variable
    bar_flag = value;
		if (bar_flag === 'SOG' && input_g.node().value.trim() === '') {
        SearchFor(input_g.node(), g_sug, genreG, "g");
    }
  	if (bar_flag === 'SOA' && input_a.node().value.trim() === '') {
        SearchFor(input_a.node(), a_sug, artistG, "a");
    }
}


var A_option = d3.selectAll('input[name="A_option"]');

// Listen for changes in the radio group
A_option.on('change', function () {
    // Get the selected value
    var selectedValue = d3.select('input[name="A_option"]:checked').node().value;
    // Set bar_flag based on the selected value
    switch (selectedValue) {
        case 'gen':
            setAnaFlag('G');
            break;
        case 'pop':
            setAnaFlag('P');
            break;
        case 'idx':
            setAnaFlag('I');
            break;
        default:
            break;
    }
  	svgContainer.selectAll("*").remove();
  	console.log(ana_flag)
  	AnalysisAll();
});

// Function to set the bar_flag
function setAnaFlag(value) {
    // Assuming bar_flag is a global variable
    ana_flag = value;
}



