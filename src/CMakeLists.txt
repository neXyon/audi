SET(HEADERS_LIST
	functions.hpp
	functions_from_d.hpp
	audi.hpp
	gdual.hpp
	neural_net.hpp
)

# NOTE: this dummy cpp file is here with the sole purpose of getting the headers
# inside the project files generated by CMake.
add_library(audi_dummy STATIC audi_dummy.cpp ${HEADERS_LIST})

# TODO header installation.
install(FILES ${HEADERS_LIST} DESTINATION include/audi)
