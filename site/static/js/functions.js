// Get the modal
var modal_t = document.getElementById('myModal_true');
var modal_f = document.getElementById('myModal_false');
var modal_finish = document.getElementById('myModal_finish');

// Get the button that opens the modal
var btn_t = document.getElementById("but_true");
var btn_f = document.getElementById("but_false");


var player_score_el = document.getElementById("player-score");
var ai_score_el = document.getElementById("AI-score");
var article_content = document.getElementById("article-content");


// When the user clicks the button, open the modal
btn_t.onclick = function () {
    console.log("Button True clicked");
    sendAnswer(true)
};

btn_f.onclick = function () {
    console.log("Button False clicked");
    sendAnswer(false)
};


// When the user clicks anywhere outside of the modal, close it
window.onclick = function (event) {
    if (event.target === modal_t) {
        modal_t.style.display = "none";
    }
    if (event.target === modal_f) {
        modal_f.style.display = "none";
    }
};

function sendAnswer(answer) {
    $.post('/game', {
        value: answer
    }).done(function (resFromServer) {
        console.log(resFromServer);
        var playerScore = parseInt(player_score_el.innerHTML);
        var aiScore = parseInt(ai_score_el.innerHTML);

        if (resFromServer['aiCorrect'] === true) {
            aiScore += 1;
            ai_score_el.innerHTML = aiScore.toString();
        }

        if (resFromServer['correct'] === true) {
            playerScore += 1;
            player_score_el.innerHTML = playerScore.toString();
            showPopupGoodAnswer();
        } else {
            showPopupBadAnswer();
        }

        if (resFromServer['displayPopupFinish'] === true) {
            console.log("doit afficher popup de fin");

            var playerScoreModalEl = document.getElementById("player_score_finish");
            var playerScoreText = document.createTextNode(playerScore.toString());
            playerScoreModalEl.appendChild(playerScoreText);

            var aiScoreModalEl = document.getElementById("ai_score_finish");
            var aiScoreText = document.createTextNode(aiScore.toString());
            aiScoreModalEl.appendChild(aiScoreText);

            setTimeout(showPopupFinish, 1500);
            article_content.innerHTML = ""
        } else {
            article_content.innerHTML = resFromServer['newArticleContent']
        }


    }).fail(function () {
        console.log("failed")
    });

}

function showPopupFinish() {
    modal_finish.style.display = "block";
}

function showPopupGoodAnswer() {
    modal_t.style.display = "block";
    setTimeout(function () {
        modal_t.style.display = "none";
    }, 3000);
}

function showPopupBadAnswer() {
    modal_f.style.display = "block";
    setTimeout(function () {
        modal_f.style.display = "none";
    }, 3000);
}
