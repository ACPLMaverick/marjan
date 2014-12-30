package com.plodz.cartracker;

import java.util.GregorianCalendar;

public class Globals {
	public static enum fuelType {DIESEL, PETROL, LPG};
	
	public static fuelType myFuelType;
	public static float myFuelConsumption;
	public static float DBG_updateRatio;
	public static int checkDelay;
	public static boolean showHigherPrice;
	public static float priceDiesel;
	public static float priceDieselUltimate;
	public static float pricePB95;
	public static float pricePB98;
	public static GregorianCalendar lastUpdate;
	public static float mapZoomMultiplier;
	
	public static String fuelURL = "http://www.e-petrol.pl/notowania/rynek-krajowy/ceny-stacje-paliw";
}
