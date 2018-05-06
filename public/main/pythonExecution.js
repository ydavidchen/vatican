function runPyScript(input){
    var jqXHR = $.ajax({
        type: "POST",
        url: "/login",
        async: false,
        data: { mydata: input }
    });

    return jqXHR.responseText;
}

$('#submitbutton').click( function() {
    datatosend = 'this is my matrix';
    result = runPyScript(datatosend);
    console.log('Got back ' + result);
});