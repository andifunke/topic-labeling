<?php

### Konfiguration ###

# Bitte passen Sie die folgenden Werte an, bevor Sie das Script benutzen!

# An welche Adresse sollen die Mails gesendet werden?
$strEmpfaenger = '...';

# Welche Adresse soll als Absender angegeben werden?
# (Manche Hoster lassen diese Angabe vor dem Versenden der Mail ueberschreiben)
$strFrom       = '...';

# Welchen Betreff sollen die Mails erhalten?
$strSubject    = 'PHP-Workshop Feedback';

# Zu welcher Seite soll als "Danke-Seite" weitergeleitet werden?
# Wichtig: Sie muessen hier eine gueltige HTTP-Adresse angeben!

$strReturnhtml = 'index.php';

# Welche(s) Zeichen soll(en) zwischen dem Feldnamen und dem angegebenen Wert stehen?
$strDelimiter  = ":\t";

### Ende Konfiguration ###

if($_POST)
{
 $strMailtext = '';
 $strMailtext .= "\nEinleitung:\n";
 $strMailtext .= $_POST[einleitung1] . "\n";
 $strMailtext .= $_POST[einleitung2] . "\n";
 $strMailtext .= $_POST[einleitung3] . "\n";
 $strMailtext .= "\nClient-Server:\n";
 $strMailtext .= $_POST[clientserver1] . "\n";
 $strMailtext .= $_POST[clientserver2] . "\n";
 $strMailtext .= $_POST[clientserver3] . "\n";
 $strMailtext .= "\nPHP Vortrag:\n";
 $strMailtext .= $_POST[phpvortrag1] . "\n";
 $strMailtext .= $_POST[phpvortrag2] . "\n";
 $strMailtext .= $_POST[phpvortrag3] . "\n";
 $strMailtext .= "\nPHP Übung:\n";
 $strMailtext .= $_POST[phpuebung1] . "\n";
 $strMailtext .= $_POST[phpuebung2] . "\n";
 $strMailtext .= $_POST[phpuebung3] . "\n";
 $strMailtext .= "\nAnwendungsbeispiele Vortrag:\n";
 $strMailtext .= $_POST[anwendungvortrag1] . "\n";
 $strMailtext .= $_POST[anwendungvortrag2] . "\n";
 $strMailtext .= $_POST[anwendungvortrag3] . "\n";
 $strMailtext .= "\nAnwendungsbeispiele Übung:\n";
 $strMailtext .= $_POST[anwendunguebung1] . "\n";
 $strMailtext .= $_POST[anwendunguebung2] . "\n";
 $strMailtext .= $_POST[anwendunguebung3] . "\n";
 $strMailtext .= "\nDiskussionsrunde:\n";
 $strMailtext .= $_POST[diskussion1] . "\n";
 $strMailtext .= $_POST[diskussion2] . "\n";
 $strMailtext .= $_POST[diskussion3] . "\n";
 $strMailtext .= "\nFeedback-Formular:\n";
 $strMailtext .= $_POST[feedback1] . "\n";
 $strMailtext .= $_POST[feedback2] . "\n";
 $strMailtext .= $_POST[feedback3] . "\n";
 $strMailtext .= "\nInsgesamt:\n";
 $strMailtext .= $_POST[insgesamt1] . "\n";
 $strMailtext .= $_POST[insgesamt2] . "\n";
 $strMailtext .= $_POST[insgesamt3] . "\n";

  if(get_magic_quotes_gpc())
 {
  $strMailtext = stripslashes($strMailtext);
 }

 mail($strEmpfaenger, $strSubject, $strMailtext, "From: ".$strFrom)
  or die("Das Formular konnte nicht versendet werden.");
 header("Location: $strReturnhtml");
 exit;
}

/* include 'index.php'; */
?>