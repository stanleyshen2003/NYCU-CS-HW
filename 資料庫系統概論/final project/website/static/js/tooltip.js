const tooltips =document.querySelectorAll('.tooltip');
const alltooltips =document.querySelector('.all-tooltips');
let timeoutId;

window.addEventListener("DOMContentLoaded",contentPosition);
window.addEventListener("resize",contentPosition);

function contentPosition(){
    tooltips.forEach(tooltip=>{
        const pin = tooltip.querySelector('.pin');
        const content = tooltip.querySelector('.tooltip-content');
        content.style.top = pin.offsetTop+'px';

    })
}
contentPosition();
alltooltips.addEventListener('mousemove',()=>{
    alltooltips.style.paddingBottom='120px';
}) 
alltooltips.addEventListener('mouseleave',()=>{
    timeoutId = setTimeout(()=>{
        alltooltips.style.paddingBottom='60px';
    },100)  
}) 
tooltips.forEach(tooltip=>{
    const pin = tooltip.querySelector('.pin');
    const content = tooltip.querySelector('.tooltip-content');   
    pin.addEventListener('mouseover',()=>{
        tooltip.classList.add('active');
    }) 
    pin.addEventListener('mouseleave',()=>{
        timeoutId = setTimeout(()=>{
            tooltip.classList.remove('active');
        },100)  
        
    }) 
    content.addEventListener('mousemove',()=>{
        clearTimeout(timeoutId);
        tooltip.classList.add('active');
    }) 
    content.addEventListener('mouseleave',()=>{
        timeoutId = setTimeout(()=>{
            tooltip.classList.remove('active');
        },300)  
    }) 
})