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
            clip: false
            anchors.leftMargin: 40
            anchors.rightMargin: 40
            anchors.bottomMargin: 65
            anchors.topMargin: 95

            Image {
                id: bg_image
                anchors.fill: parent
                horizontalAlignment: Image.AlignHCenter
                source: "../../images/png_images/bg_1.PNG"
                asynchronous: false
                mirror: false
                clip: false
                fillMode: Image.PreserveAspectFit

            }
        }

        Label {
            id: welcomeLabel
            x: 90
            y: 28
            color: "#6c6565"
            text: qsTr("Welcome")
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.leftMargin: 50
            anchors.rightMargin: 50
            font.pointSize: 22
            anchors.topMargin: 20
        }

        Label {
            id: welcomeLabel1
            x: 90
            y: 69
            color: "#6c6565"
            text: qsTr("to")
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.leftMargin: 50
            anchors.rightMargin: 50
            anchors.topMargin: 61
            font.pointSize: 14
        }

        CustomBtn{
            anchors.left: parent.horizontalCenter
            anchors.top: content.bottom
            anchors.bottom: parent.bottom
            anchors.topMargin: 10
            anchors.leftMargin: -100
            anchors.bottomMargin: 10
            onClicked: {
                stackView.push(Qt.resolvedUrl("./homePage-2.qml"))
            }
        }
    }

}

/*##^##
Designer {
    D{i:0;autoSize:true;formeditorZoom:0.66;height:480;width:800}D{i:3}
}
##^##*/
