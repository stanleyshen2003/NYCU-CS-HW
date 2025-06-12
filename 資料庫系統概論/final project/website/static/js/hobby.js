var hobby=document.getElementById('hobby');
var hobby1=document.getElementById('icon1');
var hobby2=document.getElementById('icon2');
var hobby3=document.getElementById('icon3');
var hobby4=document.getElementById('icon4');
var hobby5=document.getElementById('icon5');

hobby1.addEventListener("mouseover",function(event) {
      hobby.textContent="Shopping";
      hobby.style.paddingLeft='0%';
      hobby.style.marginLeft='-10%';
      document.getElementById('hobby_icon1').classList.add('act');
    },
    false
);
hobby1.addEventListener('mouseleave',()=>{
  hobby.textContent=" ";
  setTimeout(()=>{
   
      hobby.style.marginLeft='-5%';
      document.getElementById('hobby_icon1').classList.remove('act');
  },100)  
  
}) 
hobby2.addEventListener("mouseover",function(event) {
    hobby.textContent="Music";
    hobby.style.paddingLeft='23%';
    document.getElementById('hobby_icon2').classList.add('act');
  },
  false
);
hobby2.addEventListener('mouseleave',()=>{
  hobby.textContent=" ";
  setTimeout(()=>{
   
      hobby.style.marginLeft='-5%';
      document.getElementById('hobby_icon2').classList.remove('act');
  },100)  
  
}) 
hobby3.addEventListener("mouseover",function(event) {
    hobby.textContent="Read";
    hobby.style.paddingLeft='45.5%';
    document.getElementById('hobby_icon3').classList.add('act');
  },
  false
);
hobby3.addEventListener('mouseleave',()=>{
  hobby.textContent=" ";
  setTimeout(()=>{
   
      hobby.style.marginLeft='-5%';
      document.getElementById('hobby_icon3').classList.remove('act');
  },100)  
  
}) 
hobby4.addEventListener("mouseover",function(event) {
    hobby.textContent="Game";
    hobby.style.paddingLeft='66%';
    document.getElementById('hobby_icon4').classList.add('act');
  },
  false
);
hobby4.addEventListener('mouseleave',()=>{
  hobby.textContent=" ";
  setTimeout(()=>{
   
      hobby.style.marginLeft='-5%';
      document.getElementById('hobby_icon4').classList.remove('act');
  },100)  
  
}) 
hobby5.addEventListener("mouseover",function(event) {
    hobby.textContent="Eat";
    hobby.style.paddingLeft='92%';
    document.getElementById('hobby_icon5').classList.add('act');
  },
  false
);
hobby5.addEventListener('mouseleave',()=>{
  hobby.textContent=" ";
  setTimeout(()=>{
   
      hobby.style.marginLeft='-5%';
      document.getElementById('hobby_icon5').classList.remove('act');
  },100)  
  
}) 