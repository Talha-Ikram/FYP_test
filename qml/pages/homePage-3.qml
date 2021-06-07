import QtQuick 2.0
import QtQuick.Controls 2.15
import "../controls"

Item {
    Rectangle {
        id: bg
        color: "#2c313c"
        anchors.fill: parent

        Rectangle {
            id: content
            color: "#ffffff"
            radius: 20
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            clip: true
            anchors.leftMargin: 40
            anchors.rightMargin: 40
            anchors.bottomMargin: 65
            anchors.topMargin: 95

            Image {
                id: bg_image
                x: 0
                y: 0
                anchors.fill: parent
                horizontalAlignment: Image.AlignHCenter
                source: "../../images/png_images/bg_3.PNG"
                asynchronous: false
                mirror: false
                anchors.rightMargin: 0
                anchors.bottomMargin: 0
                anchors.leftMargin: 0
                anchors.topMargin: 0
                clip: false
                fillMode: Image.PreserveAspectFit
            }
        }

        Label {
            id: welcomeLabel
            x: 90
            y: 28
            color: "#6c6565"
            text: qsTr("See it in action!")
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.leftMargin: 50
            anchors.rightMargin: 50
            font.pointSize: 22
            anchors.topMargin: 31
        }

        CustomBtn{
            width: 220
            anchors.left: parent.horizontalCenter
            anchors.top: content.bottom
            anchors.bottom: parent.bottom
            btnText: "Next"
            anchors.topMargin: 15
            anchors.leftMargin: -100
            anchors.bottomMargin: 15
            onClicked: {
                stackView.push(Qt.resolvedUrl("./homePage-4.qml"))
            }
        }
    }

}






/*##^##
Designer {
    D{i:0;autoSize:true;formeditorZoom:0.33;height:480;width:640}
}
##^##*/
