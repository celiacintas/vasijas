var started = false;
var canvas, context;
var stampId = '';
var lastColor = 'black';
var lastStampId = '';

function init() {
	canvas = $('#imageView').get(0);
	context = canvas.getContext('2d');
	
	// Auto-adjust canvas size to fit window.
	canvas.width  = 512 //window.innerWidth - 75;
	canvas.height = 512 //window.innerHeight - 75;
	//$('#container').get(0).addEventListener('mousemove', onMouseMove, false);
	canvas.addEventListener('mousedown', onMouseMove, false);
	canvas.addEventListener('click', onClick, false);
	context.lineWidth = 20;
	$('#save').get(0).addEventListener('click', function(e) { onSave(); }, false);
    $('#pc').get(0).addEventListener('click', function(e) { changeImage(); }, false);
    //$('#clear').get(0).addEventListener('click', function(e) { clear(); }, false);
	context.fillStyle = 'rgba(1,1,1)';
	context.fill();
	
  
	
}

function changeImage(){
    let img1 = "static/image/MB_14.png";
    let img2 = "static/image/CC_02_7.png";
    if ($('#imgid').get(0).src.split('/')[5] == "CC_02_7.png"){
        $('#imgid').get(0).src = img1
    }
    else{
        $('#imgid').get(0).src = img2
    }
    
}

function clear(){
    context.clearRect(0, 0, canvas.width, canvas.height);
}

function onMouseMove(ev) {
	var x, y;
		
	// Get the mouse position.
	if (ev.layerX >= 0) {
		// Firefox
		x = ev.layerX - 50;
		y = ev.layerY - 5;
	}
	else if (ev.offsetX >= 0) {
		// Opera
		x = ev.offsetX - 50;
		y = ev.offsetY - 5;
	}
	
	x = ev.offsetX;
	y = ev.offsetY;

	if (!started) {
		started = true;

		context.beginPath();
		context.moveTo(x, y);		
	}
	else {
		context.lineTo(x, y);
		context.stroke();
	}
	
	$('#stats').text(x + ', ' + y);
}

function onClick(e) {
	if (stampId.length > 0) {
		context.drawImage($(stampId).get(0), e.pageX , e.pageY, 80, 80);
	}
}

function onColorClick(color) {
	// Start a new path to begin drawing in a new color.
	context.closePath();
	context.beginPath();
	
	// Select the new color.
	context.strokeStyle = color;
	
	// Highlight selected color.
	var borderColor = 'white';
	if (color == 'white' || color == 'yellow') {
		borderColor = 'black';
	}
	
	$('#' + lastColor).css("border", "0px dashed white");
	$('#' + color).css("border", "1px dashed " + borderColor);
	
	// Store color so we can un-highlight it next time around.
	lastColor = color;
}

function onFill() {
	// Start a new path to begin drawing in a new color.
	context.closePath();
	context.beginPath();

	context.fillStyle = context.strokeStyle;
	context.fillRect(0, 0, canvas.width, canvas.height);
}

function onStamp(id) {
	// Update the stamp image.
	stampId = '#' + id;

    if (lastStampId == stampId) {
        // User clicked the selected stamp again, so deselect it.
        stampId = '';
    }

	$(lastStampId).css("border", "0px dashed white");
	$(stampId).css("border", "1px dashed black");
	
	// Store stamp so we can un-highlight it next time around.
	lastStampId = stampId;	
}

function onSave() {
	console.log('SAVE')
	var img = canvas.toDataURL("image/jpg");
    const fd = new FormData;
    fd.append('files', img);

    $.ajax({
            type: 'POST',
            url: '/sendImage',
            data: fd,
            async: false,
            cache: false,
            contentType: false,
            processData: false,
            success: function () {
                $('#result').empty();
                $('#result').append( "<img src='/static/image/result_land.png?"+performance.now()+"' />" );
                //alert('Form Submitted!');
                
            },
            error: function(){
                alert("error in ajax form submission");
            }
    });
	//document.write('<img src="' + img + '"/>');
}