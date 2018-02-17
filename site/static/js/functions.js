// Get the modal
var modal_t = document.getElementById('myModal_true');
var modal_f = document.getElementById('myModal_false');
// Get the button that opens the modal
var btn_t = document.getElementById("but_true");
var btn_f = document.getElementById("but_false");


// When the user clicks the button, open the modal
btn_t.onclick = function() {
    modal_t.style.display = "block";
    setTimeout(function() {modal_t.style.display = "none";},3000);
}

btn_f.onclick = function() {
    modal_f.style.display = "block";
    setTimeout(function() {modal_f.style.display = "none";},3000);

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

