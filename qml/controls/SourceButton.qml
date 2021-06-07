import QtQuick 2.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.15

Button{
    id: btnToggle

    //Custom Properties
    property color btnColorDefault: "#24a0ed"
    property color btnColorMouseOver: "#23272E"
    property color btnColorClicked: "#00a1f1"
    property string btnText: "Select this source"
    property color textColor: "#FFFFFF"
    font.pointSize: 10

    QtObject{
        id: internal

        property var dynamicColor: if(btnToggle.down){
                                       btnToggle.down ? btnColorClicked : btnColorDefault
                                   } else{
                                       btnToggle.hovered ? btnColorMouseOver : btnColorDefault
                                   }
    }


    implicitWidth: 150
    implicitHeight: 30

    background: Rectangle{
        id: bgBtn
        color: internal.dynamicColor
        radius: 20


        Text {
            id: buttonText
            color: textColor
            text: "Select this source"
            anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter
            font.pointSize: 11
        }
    }
}


