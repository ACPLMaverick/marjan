package mainPackage.View;

import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JPanel;

// TODO: Auto-generated Javadoc
/**
 * Klasa dziedziczaca po UserMenu reprezentuje interfejs graficzny menu uzytkownika.
 */
public class UserMenuUser extends UserMenu {
	
	/**
	 * Tworzy nowe okno menu uzytkownika.
	 *
	 * @param filmTitles lista tytulow filmow dla konstruktora klasy bazowej.
	 */
	public UserMenuUser(ArrayList<String> filmTitles)
	{
		super(filmTitles);
	}
}
