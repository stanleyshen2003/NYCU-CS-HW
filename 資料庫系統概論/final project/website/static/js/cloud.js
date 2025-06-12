const text = 'Self-disciplined Hardworking Organised Efficient Resourceful  Logic Rational Integrity Observant Patient',
    lines = text.split(/[,\. ]+/g),
    data = lines.reduce((arr, word) => {
        let obj = Highcharts.find(arr, obj => obj.name === word);
        if (obj) {
            obj.weight += 1;
        } else {
            obj = {
                name: word,
                weight: 1
            };
            arr.push(obj);
        }
        return arr;
    }, []);
var option2={
    series: [{
        type: 'wordcloud',
        data
    }],
    title: {
        text: ''
    },
    credits:{
        enabled:false
    },
    chart:{
        backgroundColor: '#f6faff'
    },
    tooltip: {
        enabled:false
    }
    
};
Highcharts.chart('container2',option2);
