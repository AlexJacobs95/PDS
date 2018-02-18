// Get the modal
var modal_t = document.getElementById('myModal_true');
var modal_f = document.getElementById('myModal_false');
var modal_finish = document.getElementById('myModal_finish');

// Get the button that opens the modal
var btn_t = document.getElementById("but_true");
var btn_f = document.getElementById("but_false");


var player_score = document.getElementById("player-score");
var article_content = document.getElementById("article-content");


// When the user clicks the button, open the modal
btn_t.onclick = function () {
    //modal_t.style.display = "block";
    //setTimeout(function() {modal_t.style.display = "none";},3000);
    console.log("Button True clicked")
    sendAnswer(true)


}

btn_f.onclick = function () {
    console.log("Button False clicked")
    sendAnswer(false)

}


// When the user clicks anywhere outside of the modal, close it
window.onclick = function (event) {
    if (event.target == modal_t) {
        modal_t.style.display = "none";
    }
    if (event.target == modal_f) {
        modal_f.style.display = "none";
    }
}

function sendAnswer(answer) {
    $.post('/game', {
        value: answer
    }).done(function (resFromServer) {
        console.log(resFromServer);
        var score = parseInt(player_score.innerHTML);
        if (resFromServer['displayPopupFinish']== true){
            console.log("doit afficher popup de fin");
            var paragraph = document.getElementById("player_score_finish");
            var text = document.createTextNode((score + 1).toString());
            paragraph.appendChild(text);
            showPopupFinish();
        }
        else if (resFromServer['correct'] == true) {
            player_score.innerHTML = (score + 1).toString();

        }

        article_content.innerHTML = resFromServer['newArticleContent']


    }).fail(function () {
        console.log("failed")

    });

}

function showPopupFinish() {
    modal_finish.style.display="block";
}


