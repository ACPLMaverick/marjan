package mainPackage.View;

import java.util.ArrayList;

public class UserMenuAdmin extends UserMenu {
	
	public UserMenuAdmin(ArrayList<String> filmTitles)
	{
		super(filmTitles);
		this.userTitle.setText("  Panel administratora");
	}
}
