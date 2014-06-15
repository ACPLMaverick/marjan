package mainPackage.View;

import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;


public class MyKeyAdapter extends KeyAdapter {
	public void keyTyped(KeyEvent e)
	{
		char myChar = e.getKeyChar();
		if(((myChar != '0' && 
				myChar != '1' && 
				myChar != '2' &&
				myChar != '3' &&
				myChar != '4' &&
				myChar != '5' &&
				myChar != '6' &&
				myChar != '7' &&
				myChar != '8' &&
				myChar != '9' &&
				myChar != '.' &&
				myChar != '-' )) && (myChar != KeyEvent.VK_BACK_SPACE || myChar != KeyEvent.VK_ENTER))
		{
			e.consume();
		}
	}
}