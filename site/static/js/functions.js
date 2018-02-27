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
    sendAnswer(true)
};

btn_f.onclick = function () {
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

var popupTime = 1500;

function sendAnswer(answer) {
    $.post('/game', {
        value: answer
    }).done(function (resFromServer) {
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

            var victory_defeat = document.getElementById("victory_or_defeat");
            if (playerScore > aiScore) {
                victory_defeat.innerHTML = "VICTOIRE !";
                victory_defeat.style.color = "green";
            } else if (aiScore > playerScore) {
                victory_defeat.innerHTML = "DÉFAITE !";
                victory_defeat.style.color = "red";
            } else {
                victory_defeat.innerHTML = "ÉGALITÉ !";
                victory_defeat.style.color = "blue";

            }

            setTimeout(function () {
                showPopupFinish();
                article_content.innerHTML = "Fin du jeu!"
            }, 1500);

        } else {
            setTimeout(function () {
                article_content.innerHTML = resFromServer['newArticleContent'];
                article_counter = article_counter + 1;
                article_counter_text.innerHTML = "Article ".concat(article_counter.toString()).concat(" sur 5");
            }, popupTime)
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
    }, popupTime);
}
function toggleShow() {
    var x = document.getElementById("copyPasteZone");
    var btn = document.getElementById("buttonShowCopyPasteZone");
    if (x.style.display === "none") {
        ClearZoneCopyPaste();
        x.style.display = "block";
        btn.style.display = 'none';
    }
}

function SendTextToServer() {
    var textToAnalyse = document.getElementById('textToAnalyse');
    textToAnalyse = textToAnalyse.value;
    if (textToAnalyse === ''){
        alert('Veuillez entrer un article en Anglais. SVP.');
    }
    else{
        sendText(textToAnalyse);
    }
}

function sendText(textToAnalyse) {
    $.post('/index', {
        value: textToAnalyse
    }).done(function (resFromServer) {
        ShowPopupAnalyseResult(resFromServer["result"]);
    }).fail(function () {
        console.log("failed")
    });
}

function CopyPasteZoneHide() {
    var x = document.getElementById("copyPasteZone");
    var btn = document.getElementById("buttonShowCopyPasteZone");
    x.style.display = 'none';
    btn.style.display = 'block';
    ClosePopup();
}

function ClosePopup() {
    var popup = document.getElementById('myModal_analyse');
    ClearZoneCopyPaste();
    popup.style.display = 'none';
}

function ClearZoneCopyPaste() {
    var textToAnalyse = document.getElementById('textToAnalyse');
    textToAnalyse.value = '';
}

function ShowPopupAnalyseResult(result) {

    var text = document.getElementById("resultText");
    var popup = document.getElementById('myModal_analyse');
    if (result){
        text.innerHTML = "VRAI !";
        text.style.color = "green";
    }
    else{
        text.innerHTML = "FAUX !";
        text.style.color = "red";
    }
    popup.style.display = 'block';

}