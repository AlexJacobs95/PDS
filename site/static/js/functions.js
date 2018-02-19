// Get the modal
var modal_finish = document.getElementById('myModal_finish');
var modal_answer = document.getElementById('myModal_answer');

// Get the button that opens the modal
var btn_t = document.getElementById("but_true");
var btn_f = document.getElementById("but_false");


var player_score_el = document.getElementById("player-score");
var ai_score_el = document.getElementById("AI-score");
var article_content = document.getElementById("article-content");

var article_counter_text = document.getElementById("article-counter");
var article_counter = 1;

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
            var img_robot_answer = document.getElementById("robot_answer_img");
            img_robot_answer.src = "../static/assets/ai_good_answer.png";
        }
        else {
            var img_robot_answer = document.getElementById("robot_answer_img");
            img_robot_answer.src = "../static/assets/ai_bad_answer.png";
        }

        if (resFromServer['correct'] === true) {
            playerScore += 1;
            player_score_el.innerHTML = playerScore.toString();
            var img_player_answer = document.getElementById("player_answer_img");
            img_player_answer.src = "../static/assets/good_answer.jpeg";

        } else {
            var img_player_answer = document.getElementById("player_answer_img");
            img_player_answer.src = "../static/assets/bad_answer.jpeg";
        }
        showPopupAnswer();

        if (resFromServer['displayPopupFinish'] === true) {
            console.log("doit afficher popup de fin");

            var playerScoreModalEl = document.getElementById("player_score_finish");
            var playerScoreText = document.createTextNode(playerScore.toString());
            playerScoreModalEl.appendChild(playerScoreText);

            var aiScoreModalEl = document.getElementById("ai_score_finish");
            var aiScoreText = document.createTextNode(aiScore.toString());
            aiScoreModalEl.appendChild(aiScoreText);

            setTimeout(showPopupFinish, 1500);
            article_content.innerHTML = "Fin du jeu!"
        } else {
            article_content.innerHTML = resFromServer['newArticleContent']
            article_counter = article_counter +1;
            article_counter_text.innerHTML = "Article ".concat(article_counter.toString()).concat(" sur 5");
        }


    }).fail(function () {
        console.log("failed")
    });

}

function showPopupFinish() {
    modal_finish.style.display = "block";
}

function showPopupAnswer() {
    myModal_answer.style.display = "block";
    setTimeout(function () {
        modal_answer.style.display = "none";
    }, 1500);
}
