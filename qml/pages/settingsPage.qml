import QtQuick 2.0
import QtQuick.Controls 2.15
import "../controls"
import "../"

Item {
    Rectangle {
        id: bg
        color: "#2c313c"
        anchors.fill: parent
        Component.onCompleted: backend.getValue(0)

        Rectangle {
            id: content
            x: 70
            y: 56
            width: 660
            height: 364
            color: "#1d2b33"
            radius: 10
            anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter



            Label {
                id: label
                color: "#cdcecf"
                text: qsTr("Configure your settings here. Do not forget to click on save!")
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                font.pointSize: 10
                anchors.bottomMargin: 320
                anchors.rightMargin: 30
                anchors.leftMargin: 20
                anchors.topMargin: 20
            }

            Rectangle {
                id: divider
                color: "#898d8f"
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                anchors.topMargin: 50
                anchors.bottomMargin: 312
                anchors.rightMargin: 10
                anchors.leftMargin: 10
            }

            CustomBtn {
                id: button
                x: 275
                y: 264
                text: qsTr("")
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 50
                btnText: "Save"
                anchors.horizontalCenter: parent.horizontalCenter
                onClicked: {
                    backend.saveValue(slider.value)
                }
            }

            Label {
                id: thresh
                color: "#cdcecf"
                text: qsTr("Distance Threshold (90-170):")
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: divider.bottom
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 230
                anchors.topMargin: 66
                anchors.leftMargin: 140
                anchors.rightMargin: 350
                font.pointSize: 10
            }

            Slider {
                id: slider
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                anchors.topMargin: 106
                anchors.rightMargin: 140
                anchors.leftMargin: 330
                anchors.bottomMargin: 218
                font.pointSize: 8
                stepSize: 1
                to: 170
                from: 90
                value: 100
                onValueChanged: {
                    sliderValue.text = slider.value.toFixed(0)
                }
            }

            Label {
                id: sliderValue
                color: "#cdcecf"
                text: qsTr("100")
                anchors.left: slider.right
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                anchors.rightMargin: 108
                anchors.leftMargin: 6
                anchors.bottomMargin: 231
                anchors.topMargin: 120
            }
        }
    }
    Timer{
        id: timer1
        interval: 1000
        running: false
        onTriggered: {
            labelBottomStatus.text = "Ready!"
            labelBottomStatus.color= "#068643"
        }
    }

    Connections{
        target: backend

        function onSaveValueSignal(thresh_value){
            labelBottomStatus.text = "Saving..."
            labelBottomStatus.color= "#FF8C00"
            timer1.running = true

        }

        function onGetValueSignal(value){
            slider.value = value
        }
    }

}

/*##^##
Designer {
    D{i:0;autoSize:true;formeditorZoom:0.9;height:480;width:800}
}
##^##*/
