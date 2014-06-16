package mainPackage.Model;

import java.util.ArrayList;
import java.util.Date;

// TODO: Auto-generated Javadoc
/**
 * 
 * Reprezentuje ca³y repertuar, przechowuje kolekcjê seansów
 * Automatycznie synchronizuje siê z baz¹ danych.
 *
 */
public class Repertoire {
	
	private ArrayList<Seance> seanceList;
	private ArrayList<Film> filmList;
	private DBController myDBController;
	
	/** Pole typu logicznego okreslajace czy nawiazane jest polaczenie z baza danych. */
	public boolean connectedMode = true;
	
	/**
	 * Tworzy nowy obiekt typu Repertoire.
	 */
	public Repertoire()
	{
		this.seanceList = new ArrayList<Seance>();
		this.myDBController = new DBController();
		this.seanceList = this.myDBController.getWholeRepertoire();
		this.filmList = this.myDBController.getAllFilms();
	}
	
	/**
	 * Dodaje nowy seans do listy seansow oraz jesli istnieje polaczenie z baza, takze do bazy danych.
	 *
	 * @param seance seans.
	 */
	public void add(Seance seance)
	{
		seanceList.add(seance);
		if(connectedMode) myDBController.addSeance(seance);
	}
	
	/**
	 * Usuwa seans o podanym indeksie z listy seansow oraz jesli istnieje polaczenie z baza, takze z bazy danych.
	 *
	 * @param i indeks w liscie seansow.
	 */
	public void delete(int i)
	{
		Seance toDelete = seanceList.get(i);
		seanceList.remove(i);
		if(connectedMode) myDBController.deleteSeance(toDelete.getDateAsString());
	}
	
	/**
	 * Aktualizuje seans w bazie danych jesli istnieje polaczenie z baza.
	 *
	 * @param seance seans.
	 * @param i indeks w liscie seansow.
	 */
	public void update(Seance seance, int i)
	{
		if(connectedMode) 
		{
			myDBController.updateSeance(seance, seanceList.get(i).getDateAsString());
			this.seanceList = this.myDBController.getWholeRepertoire();
		}
		else
		{
			System.out.println("update unavailable being not in connected mode!");
		}
	}
	
	/**
	 * Pobiera seans o podanym indeksie z listy seansow.
	 *
	 * @param i indeks w liscie seansow.
	 * @return Seans o podanym indeksie.
	 */
	public Seance get(int i)
	{
		return seanceList.get(i);
	}
	
	/**
	 * Zwraca cala liste seansow.
	 *
	 * @return Liste seansow.
	 */
	public ArrayList<Seance> get()
	{
		return seanceList;
	}
	//////////////////////////////////////
	
	/**
	 * Zwraca cala liste filmow.
	 *
	 * @return Liste filmow.
	 */
	public ArrayList<Film> getFilms()
	{
		return filmList;
	}

	/**
	 * Dodaje nowy film do listy filmow oraz jesli istnieje polaczenie z baza, takze do bazy danych.
	 *
	 * @param film film.
	 */
	public void addFilm(Film film)
	{
		filmList.add(film);
		if(connectedMode) myDBController.addFilm(film);
	}
	
	/**
	 * Usuwa film o podanym indeksie z listy filmow oraz jesli istnieje polaczenie z baza, takze z bazy danych.
	 *
	 * @param i indeks w liscie filmow.
	 */
	public void deleteFilm(int i)
	{
		Film toDelete = filmList.get(i);
		filmList.remove(i);
		if(connectedMode) myDBController.deleteFilm(toDelete.getTitle());
	}
	
	/**
	 * Aktualizuje film w bazie danych jesli istnieje polaczenie z baza.
	 *
	 * @param film film.
	 * @param i indeks w liscie filmow.
	 */
	public void updateFilm(Film film, int i)
	{
		if(connectedMode) 
		{
			myDBController.updateFilm(film, filmList.get(i).getTitle());
			this.filmList = this.myDBController.getAllFilms();
		}
		else
		{
			System.out.println("update unavailable being not in connected mode!");
		}
	}
	
	/**
	 * Pobiera film o podanym indeksie z listy filmow.
	 *
	 * @param i indeks w liscie filmow.
	 * @return Film o podanym indeksie.
	 */
	public Film getFilm(int i)
	{
		return filmList.get(i);
	}
}
