import QtQuick 2.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.15

Button{
    id: btnToggle

    //Custom Properties
    property color btnColorDefault: "#24a0ed"
    property color btnColorMouseOver: "#23272E"
    property color btnColorClicked: "#00a1f1"
    property string btnText: "Let's Get Started!"

    QtObject{
        id: internal

        property var dynamicColor: if(btnToggle.down){
                                       btnToggle.down ? btnColorClicked : btnColorDefault
                                   } else{
                                       btnToggle.hovered ? btnColorMouseOver : btnColorDefault
                                   }
    }


    implicitWidth: 200
    implicitHeight: 50

    background: Rectangle{
        id: bgBtn
        color: internal.dynamicColor
        radius: 20

        Text {
            id: buttonText
            color: "#e4dede"
            text: btnText
            anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter
            font.pointSize: 16
        }
    }
}


