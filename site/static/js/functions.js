// Get the modal
var modal_t = document.getElementById('myModal_true');
var modal_f = document.getElementById('myModal_false');
// Get the button that opens the modal
var btn_t = document.getElementById("but_true");
var btn_f = document.getElementById("but_false");

// Get the <span> element that closes the modal
var span_t = document.getElementsByClassName("close")[0];
var span_f = document.getElementsByClassName("close")[0];

// When the user clicks the button, open the modal
btn_t.onclick = function() {
    modal_t.style.display = "block";
}

btn_f.onclick = function() {
    modal_f.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span_t.onclick = function() {
    console.log("ici");
    modal_t.style.display = "none";
}

span_f.onclick = function() {
    modal_f.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
    if (event.target == modal_t) {
        modal_t.style.display = "none";
    }
    if (event.target == modal_f) {
        modal_f.style.display = "none";
    }
}

function SendArticle(){
  // SENDS HERE THE CHOOSE
}
