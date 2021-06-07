import QtQuick 2.0
import QtQuick.Controls 2.15
import "../controls"
import "../"

Item {
    Rectangle {
        id: bg
        color: "#2c313c"
        anchors.fill: parent

        Rectangle {
            id: source1
            y: 104
            height: 386
            color: "#1d2b33"
            radius: 10
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.right: divider.left
            anchors.rightMargin: 54
            anchors.leftMargin: 100


            LeftMenuBtn{
                id: webcam
                width: 115
                height: 80
                text: ""
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                btnColorMouseOver: "#00000000"
                activeMenuColor: "#00000000"
                btnColorClicked: "#00000000"
                btnColorDefault: "#00000000"
                clip: false
                anchors.topMargin: 22
                iconWidth: 64
                iconHeight: 64
                btnIconSource: "../../images/svg_images/camera-video.svg"

            }

            Label {
                id: label
                height: 33
                color: "#e2e3e5"
                text: qsTr("Live CCTV Feed")
                anchors.top: webcam.top
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignVCenter
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.topMargin: 98
                font.pointSize: 18
            }

            Label {
                id: label1
                width: 205
                height: 51
                color: "#a2a8ae"
                text: qsTr("Detector will run on live feed from the available camera")
                anchors.top: label.bottom
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                wrapMode: Text.Wrap
                anchors.horizontalCenter: parent.horizontalCenter
                font.pointSize: 10
                anchors.topMargin: 30
            }

            SourceButton {
                id: btnSource1
                y: 307
                width: 220
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: label1.bottom
                anchors.bottom: parent.bottom
                textColor: "#348e17"
                anchors.topMargin: 58
                anchors.bottomMargin: 49
                anchors.rightMargin: 20
                anchors.leftMargin: 20
                btnColorClicked: "#000000"
                btnColorMouseOver: "#1e272f"
                btnColorDefault: "#1d2b33"
                btnText: "Select this source"

                onClicked: {
                    labelBottomStatus.text = "Detector running..."
                    labelBottomStatus.color = "#FF8C00"
                    timerRunWebcam.running = true
                }
            }
        }

        Rectangle {
            id: source2
            y: 104
            height: 386
            color: "#1d2b33"
            radius: 10
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: divider.right
            anchors.right: parent.right
            anchors.verticalCenterOffset: 0
            anchors.leftMargin: 55
            anchors.rightMargin: 100
            LeftMenuBtn {
                id: webcam1
                width: 116
                height: 80
                text: ""
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                activeMenuColor: "#00000000"
                btnIconSource: "../../images/svg_images/camera-reels.svg"
                btnColorDefault: "#00000000"
                btnColorClicked: "#00000000"
                iconWidth: 64
                btnColorMouseOver: "#00000000"
                clip: false
                anchors.topMargin: 22
                iconHeight: 64
            }

            Label {
                id: label2
                height: 33
                color: "#e2e3e5"
                text: qsTr("Pre-Recorded CCTV Feed")
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: webcam1.bottom
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                wrapMode: Text.WordWrap
                anchors.rightMargin: 2
                anchors.leftMargin: 0
                font.pointSize: 18
                anchors.topMargin: 18
            }

            Label {
                id: label3
                width: 206
                height: 51
                color: "#a2a8ae"
                text: qsTr("Detector will run on pre-recorded feed obtained from a CCTV camera ")
                anchors.top: label2.bottom
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                wrapMode: Text.Wrap
                anchors.horizontalCenter: parent.horizontalCenter
                font.pointSize: 10
                anchors.topMargin: 30
            }

            SourceButton {
                id: btnSource2
                width: 220
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: label3.bottom
                anchors.bottom: parent.bottom
                font.pointSize: 8
                textColor: "#348e17"
                anchors.leftMargin: 20
                btnColorDefault: "#1d2b33"
                btnColorClicked: "#000000"
                anchors.rightMargin: 20
                btnText: "Select this source"
                btnColorMouseOver: "#1e272f"
                anchors.bottomMargin: 44
                anchors.topMargin: 58

                onClicked: {
                    labelBottomStatus.text = "Detector running..."
                    labelBottomStatus.color = "#FF8C00"
                    timerRunCCTV.running = true
                }
            }
        }

        Rectangle {
            id: divider
            x: 298
            y: 140
            width: 1
            height: 1
            color: "#00000000"
            anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter
        }

    }

    Timer{
        id: timerRunCCTV
        interval: 500
        running: false
        onTriggered: {
            backend.runCCTV(1)
        }
    }

    Timer{
        id: timerRunWebcam
        interval: 500
        running: false
        onTriggered: {
            backend.runWebcam(1)
        }
    }

    Connections{
        target: backend

        function onRunCCTVSignal(something){
            labelBottomStatus.text = "Ready!"
            labelBottomStatus.color= "#068643"
        }

        function onRunWebcamSignal(something){
            labelBottomStatus.text = "Ready!"
            labelBottomStatus.color= "#068643"
        }

    }

}



/*##^##
Designer {
    D{i:0;autoSize:true;formeditorZoom:0.66;height:480;width:800}
}
##^##*/
