Docstrings
===========================

Where possible, update or add documentation at the beginning of the function you are making or changing, adding references if required. 
This is important so that other people know how to use your code and so that we can validate any methods you use. 
We are following the `NumPy Docs Style Guide <https://docs.scipy.org/doc/numpy-1.15.0/docs/howto_document.html>`_, 
but you can use any of the documentation in the code as an example. 
Add comments where it is not clear what you have done.

Example
---------

.. code-block:: python

      def foo(var1, long_var_name='hi'):
         r"""A one-line summary that does not use variable names or the
         function name. Several sentences providing an extended description. 
         Refer to variables using back-ticks, e.g. `var`.

         Parameters
         ----------
         var1 : array_like
            Array_like means all those objects -- lists, nested lists, etc. --
            that can be converted to an array.  We can also refer to
            variables like `var1`. The type above can either refer to an actual 
            Python type (e.g. ``int``), or describe the type of the variable 
            in more  detail, e.g. ``(N,) ndarray`` or ``array_like``.           
         long_var_name : {'hi', 'ho'}, optional
            Choices in brackets, default first when optional.

         Returns
         -------
         type
            Explanation of anonymous return value of type ``type``.
         out : type
            Explanation of `out`.

         Raises
         ------
         BadException
            Because you shouldn't have done that.

         References
         ----------
         .. [1] O. McNoleg, "The integration of GIS, remote sensing,
            expert systems and adaptive co-kriging for environmental habitat
            modelling of the Highland Haggis using object-oriented, fuzzy-logic
            and neural-network techniques," Computers & Geosciences, vol. 22,
            pp. 585-588, 1996.

         Examples
         --------
         These are written in doctest format, and should illustrate how to
         use the function.

         >>> a = [1, 2, 3]
         >>> print [x + 3 for x in a]
         [4, 5, 6]
         >>> print "a\n\nb"
         a
         b

         """
